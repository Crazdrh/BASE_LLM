import random
import os

# Define lists with different parts of a strategic plan.
openings = [
    "Utilize rapid mechanized assault",
    "Leverage precision air strikes",
    "Deploy stealth reconnaissance",
    "Execute coordinated artillery barrage",
    "Initiate flanking maneuvers",
    "Conduct decoy operations",
    "Deploy rapid mechanized assault",
    "Initiate precision air strikes",
    "Launch stealth reconnaissance",
    "Commence coordinated artillery barrage",
    "Engage in flanking maneuvers",
    "Begin decoy operations",
    "Execute covert insertion tactics",
    "Deploy a surprise ambush",
    "Initiate a diversionary attack",
    "Unleash a high-intensity offensive",
    "Begin a rapid breakthrough",
    "Implement an encirclement strategy",
    "Launch a preemptive strike",
    "Start a combined arms assault",
    "Mobilize elite strike forces",
    "Conduct simultaneous multi-axis assaults",
    "Initiate tactical infiltration",
    "Commence guerrilla warfare operations",
    "Execute an air-land integration plan",
    "Launch a precision-guided missile attack",
    "Deploy cyber warfare measures",
    "Initiate a psychological operations campaign",
    "Start a naval blockade",
    "Implement drone surveillance operations",
    "Activate electronic warfare measures"
]

objectives = [
    "to secure key terrain",
    "to disrupt enemy supply lines",
    "to destabilize enemy defenses",
    "to seize strategic positions",
    "to neutralize enemy command centers",
    "to gain aerial superiority",
    "to secure key terrain",
    "to disrupt enemy supply lines",
    "to neutralize enemy defenses",
    "to seize strategic positions",
    "to capture critical infrastructure",
    "to undermine enemy command and control",
    "to isolate enemy units",
    "to force enemy retreat",
    "to destabilize enemy operations",
    "to eliminate high-value targets",
    "to secure vital logistical routes",
    "to achieve operational superiority",
    "to establish a defensive perimeter",
    "to cut off enemy reinforcements",
    "to create strategic disarray",
    "to secure a foothold in enemy territory",
    "to encircle enemy forces",
    "to prevent enemy mobilization",
    "to disrupt enemy communications",
    "to force enemy surrender",
    "to dismantle enemy fortifications",
    "to secure urban centers",
    "to gain aerial dominance",
    "to weaken enemy morale",
    "to secure supply depots"
]

tactics = [
    "by coordinating ground and air units",
    "with a focus on swift mobility",
    "while exploiting enemy vulnerabilities",
    "through synchronized maneuvers",
    "by maintaining a high operational tempo",
    "with flexible rapid response teams",
    "by coordinating ground and air units",
    "through synchronized maneuvers",
    "by exploiting enemy weaknesses",
    "with rapid deployment of reinforcements",
    "via stealth infiltration",
    "with an emphasis on surprise attacks",
    "through precise timing and execution",
    "by leveraging advanced reconnaissance",
    "using high-tech intelligence assets",
    "by implementing cutting-edge tactics",
    "via continuous situational assessment",
    "by maximizing force concentration",
    "with strategic reserve deployments",
    "by exploiting the element of surprise",
    "through relentless pressure",
    "by overwhelming enemy positions",
    "with multi-domain integration",
    "by seizing initiative at every turn",
    "through effective command and control",
    "by integrating cyber and electronic warfare",
    "via rapid decision-making processes",
    "with superior mobility and agility",
    "by maintaining operational tempo",
    "through coordinated logistical support",
    "by ensuring secure communication channels"
]

def generate_strategic_plan():
    # Combine random selections from each list to form a strategic plan.
    return f"{random.choice(openings)}, {random.choice(objectives)}, {random.choice(tactics)}."

# Determine the path to your Downloads folder and ensure it exists.
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(downloads_path, exist_ok=True)

# Path for the output text file.
output_file = os.path.join(downloads_path, "strategic_plans.txt")

# Open the file and write 5000 strategic plans (one per line).
with open(output_file, "w") as f:
    for i in range(50000):
        plan = generate_strategic_plan()
        f.write(f"Plan {i+1}: {plan}\n")

print(f"Generated 5000 strategic plans in {output_file}")
