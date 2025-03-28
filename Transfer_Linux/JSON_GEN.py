import json
import random
import os

# Determine the Downloads folder path
downloads_path = os.path.join(os.path.expanduser("~"), "JSON_GEN_FILE")
os.makedirs(downloads_path, exist_ok=True)

# Load the strategic plans from the text file.
strategic_plans_file = os.path.join(downloads_path, "strategic_plans.txt")
try:
    with open(strategic_plans_file, "r") as f:
        # Each plan is expected to be on its own line. We'll remove any extraneous whitespace.
        plans = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    # If the file is not found, use a default list.
    plans = [
        "Utilizing a blend of rapid mechanized assault and precision air strikes, this plan aims to disrupt enemy supply lines while capturing strategic points.",
        "The operation leverages the element of surprise by deploying forward mechanized infantry with aerial reconnaissance to neutralize enemy defenses and secure vital terrain.",
        "This strategy focuses on decoy maneuvers and concentrated attacks, forcing the enemy to spread thin while friendly units secure critical positions.",
        "Combining stealth and aggressive forward maneuvers, the plan is designed to outflank the enemy, exploiting gaps in their defenses for a decisive victory.",
        "By coordinating synchronized air and ground assaults, this operation aims to overwhelm enemy fortifications, thereby ensuring rapid territorial control."
    ]


def generate_unit(unit_type, alliance):
    unit_id = random.randint(1000000000, 9999999999)
    x = round(random.uniform(0, 300), 2)
    y = round(random.uniform(0, 300), 2)

    if alliance == "Friendly":
        if unit_type == "Mechanized infantry":
            command = f"attack_move_unit({unit_id}, {round(random.uniform(1, 100), 2)}, {round(random.uniform(1, 100), 2)})"
        else:  # Aviation or other friendly types
            command = f"engage_target_unit({unit_id}, {random.randint(1000000000, 9999999999)}, {round(random.uniform(1, 150), 2)}, {round(random.uniform(1, 150), 2)})"
    else:  # Enemy units
        if unit_type == "Infantry":
            command = f"defend_position({unit_id}, {round(random.uniform(1, 100), 2)}, {round(random.uniform(1, 100), 2)})"
        else:  # Artillery or other enemy types
            command = f"bombard_area({unit_id}, {round(random.uniform(1, 100), 2)}, {round(random.uniform(1, 100), 2)})"

    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "alliance": alliance,
        "position": {"x": x, "y": y},
        "command": command
    }


def generate_json_data(coa_id):
    # Random number of units between 1 and 20 for friendly and enemy units.
    friendly_count = random.randint(1, 20)
    enemy_count = random.randint(1, 20)

    data = {}
    key = f"coa_id_{coa_id}"
    data[key] = {
        # Use a random strategic plan from the file for the overview field.
        "overview": random.choice(plans),
        "name": f"Plan {coa_id}",
        "task_allocation": [
            generate_unit(random.choice(["Mechanized infantry", "Aviation"]), "Friendly")
            for _ in range(friendly_count)
        ],
        "enemy_task_allocation": [
            generate_unit(random.choice(["Infantry", "Artillery"]), "Enemy")
            for _ in range(enemy_count)
        ]
    }
    return data


# Generate 50,000 JSON files and save them in the Downloads folder.
for i in range(50000):
    json_data = generate_json_data(i)
    filename = os.path.join(downloads_path, f"coa_{i}.json")
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)
