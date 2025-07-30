from torch_geometric.contrib.nn import GRBCDAttack, PRBCDAttack
from utils.attacks.losses import AdvLoss
from utils.attacks.search import SearchAttack


def get_attack(model, atk, log=False, **kwargs):
    attack = atk.split(":")

    if "PRBCDAttack" in attack[0]:
        return PRBCDAttack(model, int(attack[1]), loss=AdvLoss(), log=log, **kwargs)

    if "GRBCDAttack" in attack[0]:
        return GRBCDAttack(model, int(attack[1]), loss=AdvLoss(), log=log, **kwargs)

    if "SNES" in attack[0]:
        mutable_nodes = attack[1] if len(attack) > 1 else "load"
        return SearchAttack(model, loss=AdvLoss(), algorithm="SNES", log=log,
                            algo_params={"mutable_nodes": mutable_nodes}, **kwargs)

    return None
