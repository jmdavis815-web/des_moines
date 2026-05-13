from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import re
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:0.5b"
LEARNING_FILE = Path("learning_memory.json")

game_state = {
    "location": "Crappy apartment in Des Moines, Iowa",
    "realized_history": [],
    "canon": [
        "The player is an ordinary guy living in a run-down apartment in Des Moines, Iowa.",
        "The story opens in a nightmare where a ghost calls to the player, attacks him, and then he wakes up in bed to his alarm clock.",
        "The player begins each loop at home and must get ready for work.",
        "The living room TV reports strange events around Des Moines, missing people, and growing corruption allegations against the mayor.",
        "At work, the player's boss is aggressive and controlling.",
        "At the player's computer station, supernatural events begin subtly, including flashes of a ghost on the monitor.",
        "At the end of each day, the player dies no matter what choice is made.",
        "Deaths can happen through accidents, violence, freak events, muggings, traffic, or supernatural interference.",
        "After death, the player wakes up in bed again on the same morning with more awareness and possible clues.",
        "Each loop increases ghost, demon, and poltergeist activity.",
        "NPCs around the city can give hints that point toward missing people, hidden rituals, and the mayor's corruption.",
        "The mayor secretly leads a cult that sacrifices vulnerable people, including the homeless.",
        "The player's boss is part of the cult and is arranging the player's death.",
        "The boss killed the player outside the intended ritual order, damaging the ritual and trapping the day in a loop.",
        "The loop can be broken by exposing or stopping the mayor, but other endings are possible, including joining the cult.",
        "Only one history becomes realized in each selected event.",
        "Unchosen futures remain possible but unrealized."
    ],
    "loop_count": 0,
    "opening_done": False,
}

def generate_futures(player_action: str) -> str:
    history = "\n".join(game_state["realized_history"][-8:])
    canon = "\n".join(game_state["canon"])

    messages = [
        {
            "role": "system",
            "content": f"""
You generate possible futures for a GHS horror time-loop game.

Rules:
- Keep responses tense, grounded, and scary.
- Each event must be one to three sentences.
- The player is an ordinary man in Des Moines, Iowa, not a hero.
- The story should begin with the nightmare if it has not happened yet.
- The apartment, workday, TV news, boss, computer station, missing people, mayor, cult, and time loop are core story elements.
- Do not reveal the full cult explanation too early.
- Early clues should be small: TV reports, missing posters, strange screen flashes, odd NPC warnings, or deja vu.
- Supernatural activity should escalate over loops.
- Death may happen near the end of a day, but do not force death every turn.
- When death happens, the next realized history should support waking back up in bed.
- Output exactly in this format.

FUTURE 1:
score: 0.8
event: something happens

FUTURE 2:
score: 0.6
event: something else happens

FUTURE 3:
score: 0.4
event: a third thing happens

CANON:
{canon}
"""
        },
        {
            "role": "user",
            "content": f"""
LOCATION: {game_state["location"]}
LOOP COUNT: {game_state["loop_count"]}
OPENING NIGHTMARE DONE: {game_state["opening_done"]}

REALIZED HISTORY:
{history}

PLAYER ACTION TO EVALUATE:
{player_action}
"""
        }
    ]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 90,
                "temperature": 0.2,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def load_learning_memory() -> dict:
    if not LEARNING_FILE.exists():
        return {"patterns": {}}

    try:
        return json.loads(LEARNING_FILE.read_text())
    except json.JSONDecodeError:
        return {"patterns": {}}

def save_learning_memory(memory: dict) -> None:
    LEARNING_FILE.write_text(json.dumps(memory, indent=2))

def normalize_pattern(event: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", event.lower())
    return " ".join(cleaned.split()[:8])

def extract_tags(event: str) -> list:
    text = event.lower()
    tags = []

    keyword_tags = {
        "npc": ["boss", "neighbor", "coworker", "stranger", "homeless", "mayor"],
        "discovery": ["discover", "find", "hidden", "clue", "poster", "report"],
        "danger": ["warning", "danger", "accident", "mugging", "run over", "killed", "death"],
        "ghost": ["ghost", "apparition", "haunting", "poltergeist"],
        "demon": ["demon", "ritual", "cult", "sacrifice"],
        "corruption": ["mayor", "corruption", "cover-up", "city hall"],
        "work": ["work", "office", "computer", "screen", "boss"],
        "apartment": ["apartment", "alarm", "bed", "tv", "living room"],
        "missing_people": ["missing", "homeless", "vanished", "disappeared"],
        "time_loop": ["loop", "again", "restart", "wake up", "same morning"],
        "passive_scenery": ["wind", "light", "rain", "sirens", "static"],
    }

    for tag, keywords in keyword_tags.items():
        for keyword in keywords:
            if keyword in text:
                tags.append(tag)
                break

    if not tags:
        tags.append("general")

    return tags

def select_realized_future(futures_text: str) -> dict:
    memory = load_learning_memory()
    patterns = memory.setdefault("patterns", {})
    tags_memory = memory.setdefault("tags", {})
    futures = []
    blocks = re.split(r"FUTURE\s+\d+:", futures_text)

    for block in blocks:
        score_match = re.search(r"score:\s*([0-9.]+)", block, re.IGNORECASE)
        event_match = re.search(r"event:\s*(.+)", block, re.IGNORECASE | re.DOTALL)

        if not score_match or not event_match:
            continue

        try:
            score = float(score_match.group(1))
        except ValueError:
            continue

        event = event_match.group(1).strip()
        pattern = normalize_pattern(event)
        tags = extract_tags(event)
        
        learned = patterns.get(pattern, {})
        learned_weight = float(learned.get("weight", 0.0))
        
        tag_weight = 0.0

        for tag in tags:
            tag_data = tags_memory.get(tag, {})
            tag_weight += float(tag_data.get("weight", 0.0))

        final_score = score + learned_weight + tag_weight

        futures.append({
            "score": score,
            "learned_weight": learned_weight,
            "tag_weight": tag_weight,
            "final_score": final_score,
            "event": event,
            "tags": tags,
            "pattern": pattern,
        })

    if not futures:
        return {"score": 0.0, "event": "Reality becomes unstable, and no clear future resolves."}

    selected = max(futures, key=lambda future: future["final_score"])

    for future in futures:
        pattern = future["pattern"]
        entry = patterns.setdefault(pattern, {
            "shown": 0,
            "selected": 0,
            "ignored": 0,
            "weight": 0.0,
        })

        entry["shown"] += 1

        if future is selected:
            entry["selected"] += 1
            entry["weight"] = min(1.0, entry["weight"] + 0.05)
        else:
            entry["ignored"] += 1
            entry["weight"] = max(-0.5, entry["weight"] - 0.02)
            
        for tag in future["tags"]:
            tag_entry = tags_memory.setdefault(tag, {
                "shown": 0,
                "selected": 0,
                "ignored": 0,
                "weight": 0.0,
            })

            tag_entry["shown"] += 1

            if future is selected:
                tag_entry["selected"] += 1
                tag_entry["weight"] = min(
                    1.0,
                    tag_entry["weight"] + 0.03
                )
            else:
                tag_entry["ignored"] += 1
                tag_entry["weight"] = max(-0.5, tag_entry["weight"] - 0.01)

    save_learning_memory(memory)
    return selected

@app.route("/act", methods=["POST"])
def act():
    data = request.get_json(force=True)
    action = data.get("action", "").strip()

    if not action:
        return jsonify({"error": "Action is required"}), 400

    futures_text = generate_futures(action)
    selected = select_realized_future(futures_text)
    realized_event = selected["event"]

    game_state["realized_history"].append(f"Player: {action}")
    game_state["realized_history"].append(f"Reality: {realized_event}")
    
    if not game_state["opening_done"]:
        game_state["opening_done"] = True

    if any(word in realized_event.lower() for word in ["wake up", "alarm", "same morning", "restart"]):
        game_state["loop_count"] += 1

    return jsonify({
        "result": realized_event,
        "selector_score": selected["score"],
        "learned_weight": selected.get("learned_weight", 0.0),
        "final_score": selected.get("final_score", selected["score"]),
        "tag_weight": selected.get("tag_weight", 0.0),
        "tags": selected.get("tags", []),
        "raw_futures": futures_text,
        "state": game_state,
    })

@app.route("/state", methods=["GET"])
def state():
    return jsonify(game_state)

if __name__ == "__main__":
    app.run(debug=True, port=5000)