# powerups.py
import random

# --- BANCO DE DADOS DE RARIDADES ---
# Define a cor, chance e nome de cada raridade.
RARITY_DATA = {
    "Comum":    {"color": (128, 128, 128), "chance": 55},
    "Incomum":  {"color": (0, 200, 0),     "chance": 25},
    "Raro":     {"color": (0, 100, 255),   "chance": 10},
    "Épico":    {"color": (150, 0, 255),   "chance": 5},
    "Lendário": {"color": (255, 200, 0),   "chance": 3},
    "Mítico":   {"color": "polychromatic", "chance": 1.5},
    "Secreto":  {"color": (20, 20, 20),    "chance": 0.5}
}

# --- BANCO DE DADOS DE POWER-UPS E BÔNUS ---
# Contém os valores de bônus para cada power-up em cada raridade.
POWERUP_BONUSES = {
    "hp_up": {
        "name": "Vitalidade", "description": "+{bonus}% HP Máximo",
        "bonuses": {"Comum": 0.10, "Incomum": 0.20, "Raro": 0.35, "Épico": 0.50, "Lendário": 0.75, "Mítico": 1.00, "Secreto": 1.50}
    },
    "speed_up": {
        "name": "Velocidade", "description": "+{bonus}% Velocidade",
        "bonuses": {"Comum": 0.05, "Incomum": 0.10, "Raro": 0.15, "Épico": 0.20, "Lendário": 0.30, "Mítico": 0.40, "Secreto": 0.60}
    },
    "damage_up": {
        "name": "Dano", "description": "+{bonus}% Dano da Arma",
        "bonuses": {"Comum": 0.10, "Incomum": 0.20, "Raro": 0.30, "Épico": 0.45, "Lendário": 0.60, "Mítico": 0.80, "Secreto": 1.20}
    },
    "firerate_up": {
        "name": "Cadência", "description": "+{bonus}% Taxa de Tiro",
        "bonuses": {"Comum": 0.05, "Incomum": 0.10, "Raro": 0.15, "Épico": 0.25, "Lendário": 0.35, "Mítico": 0.50, "Secreto": 0.70}
    },
    "heal": {
        "name": "Kit Médico", "description": "Recupera {bonus}% da Vida",
        "bonuses": {"Comum": 0.25, "Incomum": 0.40, "Raro": 0.60, "Épico": 0.80, "Lendário": 1.00, "Mítico": 1.50, "Secreto": "total"}
    }
}

# Lista de todos os IDs de power-ups disponíveis para sorteio
AVAILABLE_POWERUPS = ["hp_up", "speed_up", "damage_up", "firerate_up", "heal"]

def get_random_rarity():
    """Sorteia uma raridade com base nas chances definidas."""
    rarities = list(RARITY_DATA.keys())
    chances = [data["chance"] for data in RARITY_DATA.values()]
    return random.choices(rarities, weights=chances, k=1)[0]

def generate_choices(count=3):
    """Gera uma lista de power-ups aleatórios para a tela de level up."""
    choices = []
    # Garante que não teremos power-ups repetidos na mesma tela
    chosen_ids = random.sample(AVAILABLE_POWERUPS, k=min(count, len(AVAILABLE_POWERUPS)))
    
    for powerup_id in chosen_ids:
        rarity_name = get_random_rarity()
        powerup_info = POWERUP_BONUSES[powerup_id]
        bonus_value = powerup_info["bonuses"][rarity_name]
        
        # Formata a descrição para mostrar o valor do bônus
        if isinstance(bonus_value, float):
            desc = powerup_info["description"].format(bonus=int(bonus_value * 100))
        else: # Para casos especiais como a cura "total"
            desc = "Cura Total + Imunidade (5s)"
        
        choices.append({
            "id": powerup_id,
            "name": powerup_info["name"],
            "description": desc,
            "rarity": rarity_name,
            "bonus": bonus_value
        })
    return choices

def apply_powerup(player, weapon, powerup):
    """Aplica o bônus do power-up escolhido às estatísticas do jogador/arma."""
    powerup_id = powerup["id"]
    bonus = powerup["bonus"]
    
    if powerup_id == "hp_up":
        increase = player["initial_max_health"] * bonus
        player["max_health"] += increase
        player["health"] += increase # Também cura o jogador pelo valor aumentado
    
    elif powerup_id == "speed_up":
        player["speed_multiplier"] += bonus
    
    elif powerup_id == "damage_up":
        weapon["damage_multiplier"] += bonus

    elif powerup_id == "firerate_up":
        weapon["firerate_multiplier"] += bonus
    
    elif powerup_id == "heal":
        if bonus == "total":
            player["health"] = player["max_health"]
            if powerup["rarity"] == "Secreto":
                player["immunity_timer"] = 300 # 5 segundos de imunidade
        else:
            player["health"] = min(player["max_health"], player["health"] + player["max_health"] * bonus)
            
    # Efeitos especiais da raridade Secreta
    if powerup["rarity"] == "Secreto":
        if powerup_id == "hp_up":
            player["has_regen"] = True
        if powerup_id == "damage_up":
            player["has_aoe_damage"] = True