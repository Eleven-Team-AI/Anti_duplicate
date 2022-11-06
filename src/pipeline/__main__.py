from .pipelines import pipeline_prediction, pipeline_create_embeddings, pipline_training
import sys 

def main(args):
    param_name = args[1]
    if param_name == '--help':
        print('--preprocess - parametr for create bert embedings(requier GPU)\n'
              '--train - parametr for train model\n'
              '--predict - parametr for predict group')
    if param_name == '--preprocess':
        pipeline_create_embeddings(config_file=args[2])
    if param_name == '--train':
        pipline_training(config_file=args[2])
    if param_name == '--predict':
        pipeline_prediction(config_path=args[2])

    

if __name__ == '__main__':
    if len (sys.argv) == 1:
        print('Enter parametrs for start pipline. For help use --help')
    else:
        main(sys.argv)