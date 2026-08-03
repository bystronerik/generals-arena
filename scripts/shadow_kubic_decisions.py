"""Replay Kubic's games and ask what sosipolis would do from the same view.

Matching Kubic's aggregate statistics failed repeatedly: its move histogram is
an output of its policy on the positions *it* reaches, so forcing our policy to
emit the same histogram on *our* positions makes worse moves. This asks the
positional question instead — given the exact observation Kubic had, fog and
all, what do we choose? Disagreements point at a decision rule rather than a
percentage.

Fidelity caveats: visibility is the engine's 3x3 block around owned cells
(generals/core/game.py::get_visibility); castles built during the game are not
in the replay ticks, so they read as plain; build actions are skipped as not
comparable.

    python scripts/shadow_kubic_decisions.py 24988 24989 24990

Match ids are files under competition-replays/Kubic/{win,lose,draw}/ that also
have an entry in competition-replays/Kubic/_derived_actions/. That corpus is
gitignored observational data — see AGENTS.md "Leaderboard replays".
"""
import json, os, sys, collections
from dataclasses import dataclass
from typing import List
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "bots" / "sosipolis"))
from brain import Agent

T_FOG, T_PLAIN, T_MOUNTAIN, T_CASTLE, T_GENERAL, T_STRUCT_FOG = 0,1,2,3,4,5
DIRS=[(-1,0),(1,0),(0,-1),(0,1)]

@dataclass
class Obs:
    H:int; W:int; turn:int; my_land:int; my_army:int; opp_land:int; opp_army:int
    type_grid:List[List[int]]; owner_grid:List[List[int]]; army_grid:List[List[int]]

def build_obs(rep, tick, seat, t):
    H=rep["dims"]["rows"]; W=rep["dims"]["cols"]
    owners=tick["owners"]; armies=tick["armies"]
    mount={(r,c) for r,c in rep["mountains"]}
    gens={tuple(g) for g in rep["generals"]}
    # 3x3 visibility around our owned cells
    vis=[[False]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if owners[r][c]==seat:
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        rr,cc=r+dr,c+dc
                        if 0<=rr<H and 0<=cc<W: vis[rr][cc]=True
    tg=[[0]*W for _ in range(H)]; og=[[0]*W for _ in range(H)]; ag=[[0]*W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            is_m=(r,c) in mount
            if vis[r][c]:
                tg[r][c]= T_MOUNTAIN if is_m else (T_GENERAL if (r,c) in gens else T_PLAIN)
                o=owners[r][c]
                og[r][c]= 0 if o==-1 else (1 if o==seat else 2)
                ag[r][c]= armies[r][c]
            else:
                tg[r][c]= T_STRUCT_FOG if is_m else T_FOG
    ml=sum(1 for r in range(H) for c in range(W) if owners[r][c]==seat)
    ma=sum(armies[r][c] for r in range(H) for c in range(W) if owners[r][c]==seat)
    ol=sum(1 for r in range(H) for c in range(W) if owners[r][c]==1-seat)
    oa=sum(armies[r][c] for r in range(H) for c in range(W) if owners[r][c]==1-seat)
    return Obs(H,W,t,ml,ma,ol,oa,tg,og,ag)

def target_of(obs, act):
    if act[0]!=0: return "pass"
    r,c,d=act[1],act[2],act[3]
    nr,nc=r+DIRS[d][0], c+DIRS[d][1]
    if not (0<=nr<obs.H and 0<=nc<obs.W): return "illegal"
    return {0:"neutral",1:"own",2:"enemy"}[obs.owner_grid[nr][nc]]

def run(match_id):
    base="competition-replays/Kubic"
    rep=None
    for o in ("win","lose","draw"):
        p=f"{base}/{o}/{match_id}.json"
        if os.path.exists(p): rep=json.load(open(p)); break
    der=json.load(open(f"{base}/_derived_actions/{match_id}.json"))
    seat=der["kubic_seat"]
    H=rep["dims"]["rows"]; W=rep["dims"]["cols"]
    agent=Agent(player_id=seat, H=H, W=W)
    agree=0; total=0
    disagree=collections.Counter(); ours=collections.Counter(); theirs=collections.Counter()
    for i,tk in enumerate(der["ticks"]):
        t=tk["t"]
        if t-1 >= len(rep["ticks"]): break
        obs=build_obs(rep, rep["ticks"][t-1], seat, t)
        try: mine=agent.act(obs)
        except Exception as e: mine=(1,0,0,0,0)
        k=tk.get(f"p{seat}")
        if not k: continue
        if k["kind"]=="pass":
            kact=(1,0,0,0,0)
        elif k["kind"]!="move" or "src" not in k:
            continue   # build / unresolved ticks are not comparable
        else:
            sr,sc=k["src"]; dr_,dc_=k["dst"][0]-sr, k["dst"][1]-sc
            try: di=DIRS.index((dr_,dc_))
            except ValueError: continue
            kact=(0,sr,sc,di,0)
        total+=1
        if tuple(mine)==kact: agree+=1
        else:
            kt = "pass" if k["kind"]=="pass" else k.get("target","?")
            mt = target_of(obs, mine)
            disagree[(kt,mt)]+=1; theirs[kt]+=1; ours[mt]+=1
    return agree,total,disagree,ours,theirs

if __name__=="__main__":
    ids=sys.argv[1:]
    A=T=0; D=collections.Counter(); O=collections.Counter(); K=collections.Counter()
    for mid in ids:
        a,t,d,o,k=run(mid); A+=a; T+=t; D+=d; O+=o; K+=k
        print(f"  match {mid}: agree {a}/{t} = {100*a/t:.0f}%")
    print(f"\nTOTAL agreement {A}/{T} = {100*A/T:.0f}%")
    print(f"\nOn the {T-A} ticks we differ — what Kubic took vs what we took:")
    print(f"  {'Kubic':<10}{'sosipolis':<12}{'count':>7}")
    for (kt,mt),v in D.most_common(12):
        print(f"  {kt:<10}{mt:<12}{v:>7}")
    print(f"\n  Kubic's targets on those ticks:    {dict(K)}")
    print(f"  sosipolis's targets on those ticks: {dict(O)}")
