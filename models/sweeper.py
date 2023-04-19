import wandb
import simple_pointnet_pp

if __name__ == "__main__":
    
    
    wandb.init(project="test-project", entity="final-year-project")

    sweep_configuration = {
        'method': 'bayes',
        'metric': {
            'goal': 'minimise',
            'name': 'validation loss'
            },
        
        'parameters': {
            'batch_size': {
                "distribution": "int_uniform",
                "max": 64,
                "min": 16
            },
            "epochs":{
                "distribution": "int_uniform",
                "max": 60,
                "min": 5
            },
            "learning_rate": {
                "distribution": "uniform",
                "max": 0.01,
                "min": 0.0001
            },
            "optimiser":{
                "distribution": "categorical",
                " values": ["adam", "sgd", "adagrad", "adamax", "adamw"]
            }

        }

    }

    wandb.agent("khptq5vx", function = simple_pointnet_pp.main(), count=1)
