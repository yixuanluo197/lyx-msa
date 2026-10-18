import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from scripts import smoke_test
from train_cm import configure_recurrent_backend
original=smoke_test._build_model
def build(config,device):
    model=original(config,device)
    configure_recurrent_backend(model,'aten')
    return model
smoke_test._build_model=build
raise SystemExit(smoke_test.main())
