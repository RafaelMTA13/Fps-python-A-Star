import pygame
import math
import sys
import random
import numpy as np
from collections import deque
import heapq
import numba


# --- CONFIGURAÇÕES GERAIS ---
SCREEN_WIDTH, SCREEN_HEIGHT = 0, 0; MAP_WIDTH, MAP_HEIGHT = 25, 25; CELL_SIZE = 8
FOV = math.pi / 3; NUM_RAYS = 600; MAX_DEPTH = 50; MOUSE_SENSITIVITY = 0.002
KEY_ROTATION_SPEED = 0.03

# --- SISTEMA DE IDIOMAS ---
LANGUAGES = {
    "pt": {
        "start_button": "Começar",
        "language_button": "English",
        "wave_hud": "Onda:",
        "health_hud": "Vida:",
        "level_hud": "Nível",
        "skip_wave_prompt": "Pular para a Onda:",
        "levelup_title": "NÍVEL UP!",
        "levelup_subtitle": "ESCOLHA UM POWER-UP:",
        "stats_damage": "Dano:",
        "stats_speed": "Velocidade:",
        "stats_attack_speed": "Vel. Ataque:",
        "stats_attack_speed_suffix": "/s",
        "stats_regen": "Recuperação HP:",
        "stats_regen_yes": "Sim (1 HP/s)",
        "stats_regen_no": "Não",
        "game_over_title": "GAME OVER",
        "game_over_survival_time": "Você sobreviveu por:",
        "game_over_kills": "Inimigos Derrotados:",
        "game_over_max_wave": "Onda Máxima Alcançada:",
        "game_over_restart_button": "Recomeçar",
        "game_over_exit_button": "Sair",
        "game_title": "Infinite Horde Shooter"
    },
    "en": {
        "start_button": "Start",
        "language_button": "Português",
        "wave_hud": "Wave:",
        "health_hud": "Health:",
        "level_hud": "Level",
        "skip_wave_prompt": "Skip to Wave:",
        "levelup_title": "LEVEL UP!",
        "levelup_subtitle": "CHOOSE A POWER-UP:",
        "stats_damage": "Damage:",
        "stats_speed": "Speed:",
        "stats_attack_speed": "Attack Speed:",
        "stats_attack_speed_suffix": "/s",
        "stats_regen": "HP Regen:",
        "stats_regen_yes": "Yes (1 HP/s)",
        "stats_regen_no": "No",
        "game_over_title": "GAME OVER",
        "game_over_survival_time": "You survived for:",
        "game_over_kills": "Enemies Defeated:",
        "game_over_max_wave": "Max Wave Reached:",
        "game_over_restart_button": "Restart",
        "game_over_exit_button": "Exit",
        "game_title": "Infinite Horde Shooter"
    }
}
current_language = "pt"

# --- CONFIGURAÇÕES DE BALANCEAMENTO DO JOGO ---
PLAYER_BASE_STATS = { "initial_health": 100, "speed": 0.05 }
WEAPON_BASE_STATS = { "fire_rate": 5, "damage":25 }
XP_FORMULA = lambda level: 100 * (level ** 1.5)
MONSTER_STATS = {
    "grunt":    {"health": 50,  "speed": 0.02, "damage": 0.5, "xp_reward": 50},
    "rusher":   {"health": 70,  "speed": 0.06, "damage": 0.5, "xp_reward": 70},
    "ranger":   {"health": 100, "speed": 0.02, "projectile_speed": 0.08, "projectile_damage": 8, "shoot_cooldown": 180, "xp_reward": 100},
    "poisoner": {"health": 120, "speed": 0.025,"damage": 0.5, "poison_duration": 300, "poison_dps": 1, "xp_reward": 120},
    "boss":     {"health": 500, "speed": 0.03, "damage": 1.0, "size": 4.0, "xp_reward": 5000,
                 "phase2_speed_mult": 1.5, "phase3_speed_mult": 1.5,
                 "shockwave_cooldown": 120, "summon_cooldown": 600}
}
SHOOT_COOLDOWN = 60 / WEAPON_BASE_STATS["fire_rate"]

# --- CORES ---
BLACK=(0,0,0); WHITE=(255,255,255); GRAY=(100,100,100); RED=(200,0,0); GREEN=(0,200,0); BLUE=(0,0,200)
CEILING_COLOR=(50,50,50); FLOOR_COLOR=(70,70,70); WEAPON_COLOR=(40,40,40); MUZZLE_FLASH_COLOR=(255,255,0)
GRUNT_COLOR=RED; RUSHER_COLOR=(0,255,100); RANGER_COLOR=(255,200,0); POISONER_COLOR=(150,0,255)
BOSS_COLOR=(20,20,20); SHOCKWAVE_COLOR=(255,100,0); PROJECTILE_COLOR = (255, 120, 0)
XP_BAR_COLOR = (255, 215, 0)
BOSS_TRAIL_COLOR = (120, 0, 0)


# --- CLASSES DE GERAÇÃO DE MAPA (inalteradas) ---
class Rect:
    def __init__(self,x,y,w,h): self.x1,self.y1,self.x2,self.y2=x,y,x+w,y+h
    @property
    def center(self): return(self.x1+(self.x2-self.x1)//2, self.y1+(self.y2-self.y1)//2)
    def intersects(self,o): return(self.x1<=o.x2 and self.x2>=o.x1 and self.y1<=o.y2 and self.y2>=o.y1)
class ArenaMapGenerator:
    def __init__(self,w,h): self.w,self.h,self.map=w,h,None
    def _place_obstacles(self):
        self.map=np.zeros((self.h,self.w),dtype=int); self.map[0,:]=1; self.map[-1,:]=1; self.map[:,0]=1; self.map[:,-1]=1
        for _ in range(30):
            if random.random()<0.7: self._create_wall()
            else: self._create_cross()
    def _create_wall(self):
        l=random.randint(3,5)
        for _ in range(10):
            x,y=random.randint(2,self.w-l-2),random.randint(2,self.h-2)
            if random.random()<0.5:
                if np.sum(self.map[y-1:y+2,x:x+l])==0: self.map[y,x:x+l]=1; return
            else:
                if np.sum(self.map[y:y+l,x-1:x+2])==0: self.map[y:y+l,x]=1; return
    def _create_cross(self):
        for _ in range(10):
            x,y=random.randint(2,self.w-3),random.randint(2,self.h-3)
            if np.sum(self.map[y-1:y+2,x-1:x+2])==0: self.map[y,x-1:x+2]=1; self.map[y-1:y+2,x]=1; return
    def _flood_fill(self,start):
        q=deque([start]); visited={start}; count=0
        while q:
            x,y=q.popleft(); count+=1
            for dx,dy in[(0,1),(0,-1),(1,0),(-1,0)]:
                nx,ny=x+dx,y+dy
                if (0<=nx<self.w and 0<=ny<self.h and self.map[ny,nx]==0 and (nx,ny) not in visited):
                    visited.add((nx,ny)); q.append((nx,ny))
        return count
    def generate_map(self):
        while True:
            self._place_obstacles(); empty=np.argwhere(self.map==0)
            if len(empty)==0: continue
            sy,sx=random.choice(empty); start=(sx+0.5,sy+0.5)
            if np.sum(self.map==0)==self._flood_fill((sx,sy)): return self.map.tolist(),start

# --- DADOS GLOBAIS DO JOGO ---
game_state = "main_menu"
GAME_MAP=[]; player={}; weapon={}; monsters=[]; projectiles=[]; particles=[]
current_wave=0; active_boss=None; xp_for_next_level=0
start_time_ticks = 0; final_survival_time = ""
kill_count = 0; max_wave_reached = 1

# --- INICIALIZAÇÃO ---
pygame.init(); screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN); SCREEN_WIDTH,SCREEN_HEIGHT=screen.get_size()
pygame.display.set_caption("Horde Shooter"); clock=pygame.time.Clock()
shoot_timer=0

# Fontes
font = pygame.font.Font(None, 26)
big_font = pygame.font.Font(None, 48)
small_font = pygame.font.Font(None, 20)
font_hud = pygame.font.Font(None, 36)
font_title = pygame.font.Font(None, 74)
font_button = pygame.font.Font(None, 40)
font_card_name = pygame.font.Font(None, 28)
font_card_rarity = pygame.font.Font(None, 22)
font_card_desc = pygame.font.Font(None, 18)

pygame.mouse.set_visible(False); pygame.event.set_grab(True)
show_wave_input=False; wave_input_text=""; shoot_cooldown=0; is_leveling_up=False; powerup_choices=[]; choice_rects=[]
show_stats = False

restart_button_rect = pygame.Rect(0,0,0,0); exit_button_rect = pygame.Rect(0,0,0,0)
start_button_rect = pygame.Rect(0,0,0,0); language_button_rect = pygame.Rect(0,0,0,0)

# Placeholder para o módulo de powerups
class PowerupsPlaceholder:
    def __init__(self):
        self.RARITY_DATA = { "Comum": {"color": (200, 200, 200)}, "Raro": {"color": (0, 150, 255)}, "Épico": {"color": (180, 0, 255)}, "Lendário": {"color": (255, 180, 0)}, "Secreto": {"color": (255, 50, 50)}, "polychromatic": {"color": "polychromatic"} }
    def generate_choices(self, num): return [{"name": f"Power-up {i+1}", "rarity": "Comum", "description": "A placeholder choice."} for i in range(num)]
    def apply_powerup(self, player, weapon, choice): pass

try:
    import powerups
except ImportError:
    print("Módulo 'powerups.py' não encontrado. Usando placeholder."); powerups = PowerupsPlaceholder()

def update_player_stats():
    player["speed"] = PLAYER_BASE_STATS["speed"] * (1 + player["speed_multiplier"])
    weapon["damage"] = WEAPON_BASE_STATS["damage"] * (1 + weapon["damage_multiplier"])
    weapon["fire_rate"] = WEAPON_BASE_STATS["fire_rate"] * (1 + weapon["firerate_multiplier"])
    weapon["cooldown_frames"] = max(1, 60 / weapon["fire_rate"])

def new_game(start_wave=1):
    global GAME_MAP,player,weapon,current_wave,monsters,projectiles,active_boss,game_map_np,xp_for_next_level,particles, start_time_ticks, kill_count, max_wave_reached
    map_gen=ArenaMapGenerator(MAP_WIDTH,MAP_HEIGHT); GAME_MAP,player_start=map_gen.generate_map()
    game_map_np=np.array(GAME_MAP)
    player.update({"x":player_start[0], "y":player_start[1], "angle":random.uniform(0,2*math.pi),
                   "initial_max_health":PLAYER_BASE_STATS["initial_health"], "max_health":PLAYER_BASE_STATS["initial_health"],
                   "health":PLAYER_BASE_STATS["initial_health"], "poison_timer":0, "level":1, "xp":0,
                   "speed_multiplier":0, "immunity_timer":0, "has_regen":False, "has_aoe_damage":False})
    weapon.update({"damage_multiplier":0, "firerate_multiplier":0})
    update_player_stats()
    xp_for_next_level=XP_FORMULA(player["level"])
    current_wave=start_wave -1
    monsters=[]; projectiles=[]; active_boss=None; particles.clear()
    spawn_wave(start_wave)
    current_wave = start_wave
    if current_wave > max_wave_reached: max_wave_reached = current_wave
    start_time_ticks = pygame.time.get_ticks(); kill_count = 0
    pygame.mouse.set_visible(False); pygame.event.set_grab(True)

def add_xp(amount):
    global xp_for_next_level,is_leveling_up,powerup_choices,player,paused
    player["xp"]+=amount
    if player["xp"]>=xp_for_next_level:
        player["level"]+=1; player["xp"]-=xp_for_next_level; xp_for_next_level=XP_FORMULA(player["level"])
        is_leveling_up=True; paused=True; powerup_choices=powerups.generate_choices(3)
        pygame.mouse.set_visible(True); pygame.event.set_grab(False)

def create_monster(m_type, x, y, level_scale=1):
    stats = MONSTER_STATS[m_type].copy()
    scale_factor = 1 + (level_scale - 1) * 0.05
    stats["health"] *= scale_factor
    stats["xp_reward"] = int(stats["xp_reward"] * scale_factor)
    if "damage" in stats: stats["damage"] *= scale_factor
    if "projectile_damage" in stats: stats["projectile_damage"] *= scale_factor
    base = {"x": x, "y": y, "type": m_type, "alive": True, "hit_timer": 0, "path": [], "path_timer": 0, "color": globals()[m_type.upper() + "_COLOR"]}
    base.update(stats)
    if m_type == "boss":
        boss_scale_factor = 1 + (level_scale - 1) * 0.5
        base["health"] = MONSTER_STATS["boss"]["health"] * boss_scale_factor
        base["damage"] = MONSTER_STATS["boss"]["damage"] * boss_scale_factor
        base["xp_reward"] = int(MONSTER_STATS["boss"]["xp_reward"] * boss_scale_factor)
        base.update({"max_health": base["health"], "phase": 1, "shockwave_cooldown": 0, "summon_cooldown": 0})
    return base

def spawn_wave(wave):
    global monsters, active_boss
    monsters = []; projectiles.clear()
    if wave > 0 and wave % 20 == 0:
        while True:
            mx, my = random.uniform(5, MAP_WIDTH - 5), random.uniform(5, MAP_HEIGHT - 5)
            if GAME_MAP[int(my)][int(mx)] == 0:
                boss_level = wave // 20; boss = create_monster("boss", mx, my, level_scale=boss_level)
                monsters.append(boss); active_boss = boss; return
    monster_pool = ["grunt"];
    if wave >= 3: monster_pool.append("rusher")
    if wave >= 7: monster_pool.append("ranger")
    if wave >= 12: monster_pool.append("poisoner")
    num_monsters = 5 + int(wave * 1.5)
    for _ in range(num_monsters):
        monster_type = random.choice(monster_pool)
        while True:
            mx, my = random.uniform(1, MAP_WIDTH - 2), random.uniform(1, MAP_HEIGHT - 2)
            if GAME_MAP[int(my)][int(mx)] == 0 and math.dist((player["x"], player["y"]), (mx, my)) > 5:
                monsters.append(create_monster(monster_type, mx, my, level_scale=wave)); break

def find_path_a_star(grid,start,end):
    def heuristic(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
    start,end=tuple(map(int,start)),tuple(map(int,end)); open_set=[(0,start)]; came_from={}; g_cost={}; g_cost[start]=0
    f_cost={}; f_cost[start]=heuristic(start,end); open_set_hash={start}
    while open_set:
        _,current=heapq.heappop(open_set); open_set_hash.remove(current)
        if current==end:
            path=[];
            while current in came_from: path.append(current); current=came_from[current]
            return path[::-1]
        for dx,dy in[(0,1),(0,-1),(1,0),(-1,0)]:
            neighbor=(current[0]+dx,current[1]+dy)
            if 0<=neighbor[0]<MAP_WIDTH and 0<=neighbor[1]<MAP_HEIGHT and grid[neighbor[1]][neighbor[0]]==0:
                t_g_cost=g_cost[current]+1
                if t_g_cost<g_cost.get(neighbor,float('inf')):
                    came_from[neighbor]=current; g_cost[neighbor]=t_g_cost; f_cost[neighbor]=t_g_cost+heuristic(neighbor,end)
                    if neighbor not in open_set_hash: heapq.heappush(open_set,(f_cost[neighbor],neighbor)); open_set_hash.add(neighbor)
    return []

@numba.jit(nopython=True)
def cast_rays(player_x,player_y,player_angle,game_map_np):
    rays=[]; start_angle=player_angle-FOV/2
    for i in range(NUM_RAYS):
        ray_angle=start_angle+(i/NUM_RAYS)*FOV; dist_w=0;hit_w=False;step=0.05
        eye_x,eye_y=math.cos(ray_angle),math.sin(ray_angle)
        while not hit_w and dist_w<MAX_DEPTH:
            dist_w+=step; tx=int(player_x+eye_x*dist_w); ty=int(player_y+eye_y*dist_w)
            if not(0<=tx<MAP_WIDTH and 0<=ty<MAP_HEIGHT): hit_w=True; dist_w=MAX_DEPTH
            elif game_map_np[ty][tx]==1: hit_w=True
        rays.append((dist_w,ray_angle))
    return rays

def draw_minimap():
    offset_x=SCREEN_WIDTH-MAP_WIDTH*CELL_SIZE
    pygame.draw.rect(screen, BLACK, (offset_x, 0, MAP_WIDTH * CELL_SIZE, MAP_HEIGHT * CELL_SIZE))
    for y,r in enumerate(GAME_MAP):
        for x,c in enumerate(r):
            if c==1:pygame.draw.rect(screen,GRAY,(offset_x+x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    player_mx,player_my=int(offset_x+player["x"]*CELL_SIZE),int(player["y"]*CELL_SIZE)
    pygame.draw.circle(screen,GREEN,(player_mx,player_my),3)
    pygame.draw.line(screen,GREEN,(player_mx,player_my),(player_mx+math.cos(player["angle"])*10,player_my+math.sin(player["angle"])*10),1)
    for m in monsters:
        if m["alive"]: pygame.draw.circle(screen,m["color"],(int(offset_x+m["x"]*CELL_SIZE),int(m["y"]*CELL_SIZE)),3)

def draw_sprites():
    sprites_to_draw = [m for m in monsters if m["alive"]] + projectiles
    for s in sprites_to_draw: s["distance"]=math.dist((player["x"],player["y"]),(s.get("x",0), s.get("y",0)))
    sprites_to_draw.sort(key=lambda s:s["distance"],reverse=True)
    for s in sprites_to_draw:
        dx=s["x"]-player["x"]; dy=s["y"]-player["y"]; angle=(math.atan2(dy,dx)-player["angle"]+math.pi)%(2*math.pi)-math.pi
        if not -FOV/1.5 < angle < FOV/1.5: continue
        dist=s["distance"];
        if dist<=0.1: continue
        proj_h=min(2000,SCREEN_HEIGHT/(dist*math.cos(angle)))
        sx=(math.tan(angle)/math.tan(FOV/2))*(SCREEN_WIDTH/2)+(SCREEN_WIDTH/2); sy=SCREEN_HEIGHT/2-proj_h/2; s_col=int(sx/(SCREEN_WIDTH/NUM_RAYS))
        if 0<=s_col<len(Z_BUFFER) and dist>Z_BUFFER[s_col]: continue
        if "type" in s:
            color=WHITE if s["hit_timer"]>0 else s["color"]; radius=int(proj_h/4)*s.get("size",1); pygame.draw.circle(screen,color,(sx,sy+proj_h/2),radius)
        else:
            pygame.draw.circle(screen,s["color"],(sx,sy+proj_h/2),int(proj_h/10))

def draw_weapon_and_flash(timer):
    w_w,w_h=150,120; w_x,w_y=SCREEN_WIDTH//2-w_w//2,SCREEN_HEIGHT-w_h
    pygame.draw.rect(screen,WEAPON_COLOR,(w_x,w_y,w_w,w_h))
    if timer>0:
        cx,cy=SCREEN_WIDTH//2,w_y; pts=[(cx,cy-30),(cx-15,cy-15),(cx-30,cy),(cx-15,cy+15),(cx,cy+30),(cx+15,cy+15),(cx+30,cy),(cx+15,cy-15)]
        pygame.draw.polygon(screen,MUZZLE_FLASH_COLOR,pts)

def draw_hud():
    lang = LANGUAGES[current_language]
    wave_text=font_hud.render(f"{lang['wave_hud']} {current_wave}",True,WHITE); screen.blit(wave_text,(10,10))
    health_color=RED if player["poison_timer"]>0 else GREEN
    health_text=font_hud.render(f"{lang['health_hud']} {int(player['health'])}/{int(player['max_health'])}",True,health_color); screen.blit(health_text,(10,40))
    if active_boss and active_boss["alive"]:
        bar_w=SCREEN_WIDTH*0.6; bar_h=30; bar_x=(SCREEN_WIDTH-bar_w)/2; bar_y=10
        health_ratio=max(0,active_boss["health"]/active_boss["max_health"])
        pygame.draw.rect(screen,BLACK,(bar_x,bar_y,bar_w,bar_h)); pygame.draw.rect(screen,RED,(bar_x,bar_y,bar_w*health_ratio,bar_h))
        for phase_threshold in [0.66, 0.33]:
            marker_x=bar_x+bar_w*phase_threshold
            pygame.draw.line(screen,WHITE,(marker_x,bar_y),(marker_x,bar_y+bar_h),2)
        pygame.draw.rect(screen,WHITE,(bar_x,bar_y,bar_w,bar_h),2)
    xp_ratio = player["xp"] / xp_for_next_level
    xp_bar_width = SCREEN_WIDTH * xp_ratio
    pygame.draw.rect(screen,(40,40,70),(0,SCREEN_HEIGHT-10,SCREEN_WIDTH,10))
    pygame.draw.rect(screen,XP_BAR_COLOR,(0,SCREEN_HEIGHT-10,xp_bar_width,10))
    level_text=font_hud.render(f"{lang['level_hud']} {player['level']}",True,WHITE); screen.blit(level_text,(10,SCREEN_HEIGHT-40))
    fps = clock.get_fps()
    fps_text = font_hud.render(f"FPS: {int(fps)}", True, WHITE)
    screen.blit(fps_text, (SCREEN_WIDTH - fps_text.get_width() - 10, 10))
    timer_y_pos = 10
    if active_boss and active_boss["alive"]: timer_y_pos = 45
    elapsed_time = (pygame.time.get_ticks() - start_time_ticks) // 1000
    minutes = elapsed_time // 60; seconds = elapsed_time % 60
    time_str = f"{minutes:02}:{seconds:02}"
    time_text = font_hud.render(time_str, True, WHITE)
    screen.blit(time_text, ((SCREEN_WIDTH - time_text.get_width()) // 2, timer_y_pos))

def draw_wave_input_box():
    box_w,box_h=400,150; box_x,box_y=(SCREEN_WIDTH-box_w)//2,(SCREEN_HEIGHT-box_h)//2
    pygame.draw.rect(screen,BLACK,(box_x,box_y,box_w,box_h)); pygame.draw.rect(screen,WHITE,(box_x,box_y,box_w,box_h),2)
    prompt_text=big_font.render(LANGUAGES[current_language]["skip_wave_prompt"],True,WHITE)
    screen.blit(prompt_text,(box_x+(box_w-prompt_text.get_width())//2,box_y+20))
    input_box=pygame.Rect(box_x+50,box_y+70,box_w-100,50); pygame.draw.rect(screen,GRAY,input_box)
    input_text=font_hud.render(wave_input_text,True,WHITE)
    screen.blit(input_text,(input_box.x+10,input_box.y+10))

def draw_levelup_screen(choices):
    global choice_rects; choice_rects=[]; lang = LANGUAGES[current_language]
    overlay=pygame.Surface((SCREEN_WIDTH,SCREEN_HEIGHT),pygame.SRCALPHA); overlay.fill((0,0,0,180)); screen.blit(overlay,(0,0))
    title_text=font_title.render(lang["levelup_title"],True,XP_BAR_COLOR); screen.blit(title_text,((SCREEN_WIDTH-title_text.get_width())//2,SCREEN_HEIGHT*0.15))
    subtitle_text = big_font.render(lang["levelup_subtitle"], True, WHITE); screen.blit(subtitle_text,((SCREEN_WIDTH-subtitle_text.get_width())//2,SCREEN_HEIGHT*0.25))
    card_w,card_h=350,220; total_width=len(choices)*card_w+(len(choices)-1)*40; start_x=(SCREEN_WIDTH-total_width)//2
    for i,choice in enumerate(choices):
        card_x=start_x+i*(card_w+40); card_y=SCREEN_HEIGHT//2-card_h//2
        rect=pygame.Rect(card_x,card_y,card_w,card_h); choice_rects.append(rect)
        rarity_color=powerups.RARITY_DATA[choice["rarity"]]["color"]
        if rarity_color=="polychromatic":
            t=pygame.time.get_ticks()/500.0; r=int(127+127*math.sin(t)); g=int(127+127*math.sin(t+2)); b=int(127+127*math.sin(t+4)); rarity_color=(r,g,b)
        pygame.draw.rect(screen,(30,30,30),rect); pygame.draw.rect(screen,rarity_color,rect,3)
        if choice["rarity"]=="Secreto": pygame.draw.rect(screen,RED,rect.inflate(4,4),1)
        rarity_text_color = WHITE if choice["rarity"] == "Secreto" else rarity_color
        rarity_text=font_card_rarity.render(choice["rarity"],True,rarity_text_color)
        name_text=font_card_name.render(choice["name"],True,WHITE)
        desc_text=font_card_desc.render(choice["description"],True,GRAY)
        screen.blit(rarity_text,(rect.centerx-rarity_text.get_width()//2,rect.y+15))
        icon_placeholder_y = rect.y + 60; icon_placeholder_text = font_hud.render("[ÍCONE]", True, GRAY)
        screen.blit(icon_placeholder_text, (rect.centerx - icon_placeholder_text.get_width()//2, icon_placeholder_y))
        screen.blit(name_text,(rect.centerx-name_text.get_width()//2,rect.y+120))
        screen.blit(desc_text,(rect.centerx-desc_text.get_width()//2,rect.y+160))

def draw_stats_panel():
    lang = LANGUAGES[current_language]; lines = []
    lines.append(f"{lang['stats_damage']} {weapon['damage']:.1f}")
    lines.append(f"{lang['stats_speed']} {player['speed']:.3f}")
    lines.append(f"{lang['stats_attack_speed']} {weapon['fire_rate']:.2f}{lang['stats_attack_speed_suffix']}")
    regen_status = lang['stats_regen_yes'] if player['has_regen'] else lang['stats_regen_no']
    lines.append(f"{lang['stats_regen']} {regen_status}")
    text_surfaces = [small_font.render(line, True, WHITE) for line in lines]
    panel_height = sum(s.get_height() for s in text_surfaces) + (len(lines) * 5)
    panel_width = max(s.get_width() for s in text_surfaces) + 20
    panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA); panel_surf.fill((20, 20, 20, 180))
    panel_rect = panel_surf.get_rect(bottomright=(SCREEN_WIDTH - 10, SCREEN_HEIGHT - 20)); screen.blit(panel_surf, panel_rect)
    y_offset = panel_rect.top + 5
    for text_surface in text_surfaces:
        screen.blit(text_surface, (panel_rect.left + 10, y_offset)); y_offset += text_surface.get_height() + 5

def draw_game_over_screen():
    global restart_button_rect, exit_button_rect; lang = LANGUAGES[current_language]
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); screen.blit(overlay, (0, 0))
    game_over_text = font_title.render(lang["game_over_title"], True, RED); text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 4)); screen.blit(game_over_text, text_rect)
    time_info_text = font_hud.render(f"{lang['game_over_survival_time']} {final_survival_time}", True, WHITE); time_rect = time_info_text.get_rect(center=(SCREEN_WIDTH / 2, text_rect.bottom + 50)); screen.blit(time_info_text, time_rect)
    kill_text = font_hud.render(f"{lang['game_over_kills']} {kill_count}", True, WHITE); kill_rect = kill_text.get_rect(center=(SCREEN_WIDTH / 2, time_rect.bottom + 40)); screen.blit(kill_text, kill_rect)
    max_wave_text = font_hud.render(f"{lang['game_over_max_wave']} {max_wave_reached}", True, XP_BAR_COLOR); max_wave_rect = max_wave_text.get_rect(center=(SCREEN_WIDTH / 2, kill_rect.bottom + 40)); screen.blit(max_wave_text, max_wave_rect)
    button_w, button_h = 200, 60; button_y = max_wave_rect.bottom + 60
    restart_button_rect = pygame.Rect((SCREEN_WIDTH/2) - button_w - 20, button_y, button_w, button_h); pygame.draw.rect(screen, GREEN, restart_button_rect); pygame.draw.rect(screen, WHITE, restart_button_rect, 2)
    restart_text = font_button.render(lang["game_over_restart_button"], True, BLACK); screen.blit(restart_text, restart_text.get_rect(center=restart_button_rect.center))
    exit_button_rect = pygame.Rect((SCREEN_WIDTH/2) + 20, button_y, button_w, button_h); pygame.draw.rect(screen, RED, exit_button_rect); pygame.draw.rect(screen, WHITE, exit_button_rect, 2)
    exit_text = font_button.render(lang["game_over_exit_button"], True, BLACK); screen.blit(exit_text, exit_text.get_rect(center=exit_button_rect.center))

def draw_main_menu():
    global start_button_rect, language_button_rect; lang = LANGUAGES[current_language]
    pygame.draw.rect(screen,CEILING_COLOR,(0,0,SCREEN_WIDTH,SCREEN_HEIGHT//2))
    pygame.draw.rect(screen,FLOOR_COLOR,(0,SCREEN_HEIGHT//2,SCREEN_WIDTH,SCREEN_HEIGHT//2))
    menu_rays = cast_rays(menu_camera['x'], menu_camera['y'], menu_camera['angle'], menu_map_np)
    width_per_ray=SCREEN_WIDTH/NUM_RAYS
    for i,(dist_w,ray_angle) in enumerate(menu_rays):
        corr_dist=dist_w*math.cos(ray_angle-menu_camera["angle"]); wall_h=(SCREEN_HEIGHT/corr_dist) if corr_dist>0 else SCREEN_HEIGHT
        wall_t=int(SCREEN_HEIGHT/2-wall_h/2); wall_b=int(SCREEN_HEIGHT/2+wall_h/2)
        shading=max(0.1,1-(dist_w/MAX_DEPTH)); wall_color=(int(GRAY[0]*shading),int(GRAY[1]*shading),int(GRAY[2]*shading))
        pygame.draw.line(screen,wall_color,(i*width_per_ray,wall_t),(i*width_per_ray,wall_b),int(math.ceil(width_per_ray)))
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 150)); screen.blit(overlay, (0, 0))
    title_text = font_title.render(lang["game_title"], True, WHITE); title_rect = title_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/3)); screen.blit(title_text, title_rect)
    button_w, button_h = 250, 60
    start_button_rect = pygame.Rect(0, 0, button_w, button_h); start_button_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 20)
    pygame.draw.rect(screen, GREEN, start_button_rect); pygame.draw.rect(screen, WHITE, start_button_rect, 2)
    start_text = font_button.render(lang["start_button"], True, BLACK); screen.blit(start_text, start_text.get_rect(center=start_button_rect.center))
    language_button_rect = pygame.Rect(0, 0, button_w, button_h); language_button_rect.center = (SCREEN_WIDTH/2, start_button_rect.bottom + 30)
    pygame.draw.rect(screen, GRAY, language_button_rect); pygame.draw.rect(screen, WHITE, language_button_rect, 2)
    lang_text = font_button.render(lang["language_button"], True, WHITE); screen.blit(lang_text, lang_text.get_rect(center=language_button_rect.center))

# --- PREPARAÇÃO PARA O LOOP PRINCIPAL ---
running=True; paused=False
menu_map_gen = ArenaMapGenerator(MAP_WIDTH, MAP_HEIGHT); menu_map, menu_start = menu_map_gen.generate_map()
menu_map_np = np.array(menu_map)
menu_camera = {'x': menu_start[0], 'y': menu_start[1], 'angle': 0}

# --- LOOP PRINCIPAL ---
while running:
    for e in pygame.event.get():
        if e.type==pygame.QUIT or(e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE): running=False
        if game_state == "main_menu":
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if start_button_rect.collidepoint(e.pos): game_state = "playing"; new_game()
                elif language_button_rect.collidepoint(e.pos): current_language = "en" if current_language == "pt" else "pt"
        elif game_state == "game_over":
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if restart_button_rect.collidepoint(e.pos): game_state = "playing"; new_game()
                elif exit_button_rect.collidepoint(e.pos): running = False
        elif game_state == "playing":
            if is_leveling_up:
                if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                    for i,rect in enumerate(choice_rects):
                        if rect.collidepoint(e.pos):
                            powerups.apply_powerup(player,weapon,powerup_choices[i]); update_player_stats()
                            is_leveling_up=False; paused=False; pygame.mouse.set_visible(False); pygame.event.set_grab(True)
                continue
            if e.type==pygame.KEYDOWN:
                if show_wave_input:
                    if e.key==pygame.K_RETURN:
                        try:
                            wave_num=int(wave_input_text);
                            if wave_num>0: new_game(start_wave=wave_num)
                        except ValueError: pass
                        show_wave_input=False; paused=False; pygame.mouse.set_visible(False); pygame.event.set_grab(True)
                    elif e.key==pygame.K_BACKSPACE: wave_input_text=wave_input_text[:-1]
                    elif e.unicode.isdigit(): wave_input_text+=e.unicode
                    elif e.key==pygame.K_ESCAPE: show_wave_input=False; paused=False; pygame.mouse.set_visible(False); pygame.event.set_grab(True)
                else:
                    if e.key==pygame.K_r: new_game()
                    if e.key==pygame.K_t: show_wave_input=True; paused=True; wave_input_text=""; pygame.mouse.set_visible(True); pygame.event.set_grab(False)
                    if e.key==pygame.K_p: add_xp(xp_for_next_level)
                    if e.key==pygame.K_o: show_stats = not show_stats
    
    screen.fill(BLACK)
    if game_state == "main_menu":
        menu_camera['angle'] += 0.003; menu_camera['angle'] %= (2 * math.pi)
        draw_main_menu()
        if not pygame.mouse.get_visible(): pygame.mouse.set_visible(True); pygame.event.set_grab(False)
    
    elif game_state == "game_over":
        draw_game_over_screen()

    elif game_state == "playing":
        if paused or is_leveling_up:
            ray_casting_results=cast_rays(player["x"],player["y"],player["angle"],game_map_np); Z_BUFFER=[r[0] for r in ray_casting_results]
            draw_sprites(); draw_minimap(); draw_weapon_and_flash(0); draw_hud()
            if show_wave_input: draw_wave_input_box()
            if is_leveling_up: draw_levelup_screen(powerup_choices)
        else:
            if shoot_cooldown>0: shoot_cooldown-=1
            keys = pygame.key.get_pressed(); mouse_buttons = pygame.mouse.get_pressed(); dx_mouse, _ = pygame.mouse.get_rel()
            is_shooting = mouse_buttons[0] or keys[pygame.K_SPACE]
            if is_shooting and shoot_timer==0 and shoot_cooldown<=0:
                shoot_timer=5; shoot_cooldown=weapon["cooldown_frames"]; target_monster=None; closest_dist=float('inf')
                for m in monsters:
                    if not m["alive"]: continue
                    m["distance"] = math.dist((player['x'], player['y']), (m['x'], m['y']))
                    angle_diff=(math.atan2(m["y"]-player["y"],m["x"]-player["x"])-player["angle"]+math.pi)%(2*math.pi)-math.pi
                    if abs(angle_diff)<FOV/20:
                        dist=0;hit_w=False;step=0.1;ex,ey=math.cos(player["angle"]),math.sin(player["angle"])
                        while not hit_w and dist < m["distance"]:
                            dist+=step; tx=int(player["x"]+ex*dist); ty=int(player["y"]+ey*dist)
                            if GAME_MAP[ty][tx]==1: hit_w=True
                        if not hit_w and m["distance"] < closest_dist: target_monster=m; closest_dist=m["distance"]
                if target_monster: target_monster["health"]-=weapon["damage"]; target_monster["hit_timer"]=5
            
            player["angle"] += dx_mouse * MOUSE_SENSITIVITY
            if keys[pygame.K_LEFT]: player["angle"] -= KEY_ROTATION_SPEED
            if keys[pygame.K_RIGHT]: player["angle"] += KEY_ROTATION_SPEED
            player["angle"] %= (2 * math.pi)

            dx,dy=0,0
            if keys[pygame.K_w]: dx+=math.cos(player["angle"])*player["speed"]; dy+=math.sin(player["angle"])*player["speed"]
            if keys[pygame.K_s]: dx-=math.cos(player["angle"])*player["speed"]; dy-=math.sin(player["angle"])*player["speed"]
            if keys[pygame.K_a]: dx+=math.sin(player["angle"])*player["speed"]; dy-=math.cos(player["angle"])*player["speed"]
            if keys[pygame.K_d]: dx-=math.sin(player["angle"])*player["speed"]; dy+=math.cos(player["angle"])*player["speed"]
            new_x=player["x"]+dx; new_y=player["y"]+dy
            if GAME_MAP[int(player["y"])][int(new_x)]==0: player["x"]=new_x
            if GAME_MAP[int(new_y)][int(player["x"])]==0: player["y"]=new_y

            if player["health"]<=0:
                game_state = "game_over"; elapsed_time = (pygame.time.get_ticks() - start_time_ticks) // 1000
                minutes = elapsed_time // 60; seconds = elapsed_time % 60
                final_survival_time = f"{minutes:02}:{seconds:02}"; pygame.mouse.set_visible(True); pygame.event.set_grab(False)
            
            # --- ATUALIZAÇÃO DE INIMIGOS, PROJÉTEIS, ETC. ---
            for m in monsters:
                if m["alive"]:
                    if m["health"]<=0: m["alive"]=False; add_xp(m["xp_reward"]); kill_count += 1; continue
                    dist_to_player=math.dist((player["x"],player["y"]),(m["x"],m["y"]))
                    
                    # <<< CORREÇÃO: Lógica de IA e movimento restaurada aqui
                    if m["type"] in ["grunt","rusher","poisoner","boss"]:
                        m["path_timer"]-=1
                        if m["path_timer"]<=0: m["path_timer"]=random.randint(30,90); m["path"]=find_path_a_star(GAME_MAP,(m["x"],m["y"]),(player["x"],player["y"]))
                        if m["path"]:
                            nx,ny=m["path"][0]; dxn,dyn=nx+0.5-m["x"],ny+0.5-m["y"]; dist=math.sqrt(dxn**2+dyn**2)
                            mdx,mdy=0,0
                            if dist>0.1: mdx,mdy=(dxn/dist)*m["speed"],(dyn/dist)*m["speed"]
                            else: m["path"].pop(0)
                            new_mx,new_my=m["x"]+mdx,m["y"]+mdy
                            if GAME_MAP[int(m["y"])][int(new_mx)]==0: m["x"]=new_mx
                            if GAME_MAP[int(new_my)][int(m["x"])]==0: m["y"]=new_my
                        if dist_to_player<(m.get("size",1)*0.8) and player["immunity_timer"]<=0: 
                            player["health"]-=m["damage"]
                            if m["type"]=="poisoner": player["poison_timer"]=m["poison_duration"]

                    # <<< CORREÇÃO: Lógica do hit_timer restaurada
                    if m["hit_timer"] > 0:
                        m["hit_timer"] -= 1

            for p in projectiles[:]:
                p["x"]+=p["vx"]*p["speed"]; p["y"]+=p["vy"]*p["speed"]
                if GAME_MAP[int(p["y"])][int(p["x"])]==1: projectiles.remove(p)
                elif math.dist((player["x"],player["y"]),(p["x"],p["y"]))<0.5 and player["immunity_timer"] <= 0:
                    player["health"]-=p.get("damage", 10); projectiles.remove(p)
            
            if player["poison_timer"]>0:
                if pygame.time.get_ticks()%(1000//MONSTER_STATS["poisoner"]["poison_dps"])<20: player["health"]-=1
                player["poison_timer"]-=1
            if player["has_regen"] and player["health"] < player["max_health"]:
                if pygame.time.get_ticks() % 60 == 0: player["health"] = min(player["max_health"], player["health"] + 1)
            
            if player["immunity_timer"] > 0: player["immunity_timer"] -= 1
            if shoot_timer>0: shoot_timer-=1
            for p in particles[:]:
                p["lifetime"]-=1; p["pos"]=(p["pos"][0]+p["velocity"][0], p["pos"][1]+p["velocity"][1])
                if p["lifetime"]<=0: particles.remove(p)
            
            if not any(m["alive"] for m in monsters) and len(monsters) > 0:
                if active_boss: active_boss = None
                current_wave += 1
                if current_wave > max_wave_reached: max_wave_reached = current_wave
                spawn_wave(current_wave)
            
            pygame.draw.rect(screen,CEILING_COLOR,(0,0,SCREEN_WIDTH,SCREEN_HEIGHT//2))
            pygame.draw.rect(screen,FLOOR_COLOR,(0,SCREEN_HEIGHT//2,SCREEN_WIDTH,SCREEN_HEIGHT//2))
            ray_casting_results=cast_rays(player["x"],player["y"],player["angle"],game_map_np)
            Z_BUFFER=[r[0] for r in ray_casting_results]
            width_per_ray=SCREEN_WIDTH/NUM_RAYS
            for i,(dist_w,ray_angle) in enumerate(ray_casting_results):
                corr_dist=dist_w*math.cos(ray_angle-player["angle"]); wall_h=(SCREEN_HEIGHT/corr_dist) if corr_dist>0 else SCREEN_HEIGHT
                wall_t=int(SCREEN_HEIGHT/2-wall_h/2); wall_b=int(SCREEN_HEIGHT/2+wall_h/2)
                shading=max(0.1,1-(dist_w/MAX_DEPTH)); wall_color=(int(GRAY[0]*shading),int(GRAY[1]*shading),int(GRAY[2]*shading))
                pygame.draw.line(screen,wall_color,(i*width_per_ray,wall_t),(i*width_per_ray,wall_b),int(math.ceil(width_per_ray)))

            draw_sprites(); draw_minimap(); draw_weapon_and_flash(shoot_timer); draw_hud()
            if show_stats: draw_stats_panel()
            if show_wave_input or is_leveling_up:
                if is_leveling_up: draw_levelup_screen(powerup_choices)
                if show_wave_input: draw_wave_input_box()
            
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()