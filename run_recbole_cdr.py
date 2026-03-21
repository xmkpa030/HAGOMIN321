
import os
import argparse

from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='HAGO', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, help='gpu id')


    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_cdr(model=args.model, config_file_list=config_file_list, gpu_id=args.gpu_id)
