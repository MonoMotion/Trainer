import fire

from trainer.cli import Trainer, Utility

if __name__ == '__main__':
    fire.Fire({'trainer': Trainer, 'util': Utility})
