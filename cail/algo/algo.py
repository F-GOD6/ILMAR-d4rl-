from .ppo import PPO, PPOExpert
from .sac import SAC, SACExpert
from .gail import GAIL, GAILExpert
from .airl import AIRL, AIRLExpert
from .cail import CAIL, CAILExpert
from .twoiwil import TwoIWIL, TwoIWILExpert
from .icgail import ICGAIL, ICGAILExpert
from .trex import TREX, TREXExpert
from .drex import DREX, DREXExpert, DREXBCExpert
from .ssrr import SSRR, SSRRExpert
from .bc import BC,BCExpert
from .iswbc import ISWBC,ISWBCExpert
from .mybc import MYBC,MYBCExpert
from .demodice import DemoDICE,DemoDICEExpert
from .metaiswbc import METAISWBC,METAISWBCExpert # type: ignore
from .metademodice import MetaDemoDICE,MetaDemoDICEExpert # type: ignore
from .ilas import ILAS,ILASExpert
from .iswbcg import ISWBCG,ISWBCGExpert
# all the algorithms
ALGOS = {
    'sac': SAC,
    'ppo': PPO,
    'gail': GAIL,
    'airl': AIRL,
    'cail': CAIL,
    '2iwil': TwoIWIL,
    'icgail': ICGAIL,
    'trex': TREX,
    'drex': DREX,
    'ssrr': SSRR,
    'bc':BC,
    'iswbc': ISWBC,
    'demodice': DemoDICE,
    'mybc':MYBC,
    'metaiswbc':METAISWBC,
    'metademodice':MetaDemoDICE,
    "ilas":  ILAS,
    'iswbcg': ISWBCG,
}

# all the well-trained algorithms
EXP_ALGOS = {
    'sac': SACExpert,
    'ppo': PPOExpert,
    'gail': GAILExpert,
    'airl': AIRLExpert,
    'cail': CAILExpert,
    '2iwil': TwoIWILExpert,
    'icgail': ICGAILExpert,
    'trex': TREXExpert,
    'drex': DREXExpert,
    'drex_bc': DREXBCExpert,
    'ssrr': SSRRExpert,
    'bc':BCExpert,
    'iswbc':ISWBCExpert,
    'demodice':DemoDICEExpert,
    'mybc':MYBCExpert,
    'metaiswbc':METAISWBCExpert,
    'metademodice':MetaDemoDICEExpert,
    'iswbcg': ISWBCG,
}