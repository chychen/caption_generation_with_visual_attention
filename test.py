from core.new_solver import CaptioningSolver
from core.new_model import CaptionGenerator
from core.utils import load_coco_data


def main():

    data = load_coco_data(data_path='./data', split='val', if_train=True)
    word_to_idx = data['word_to_idx']
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=1024, n_time_step=16, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    solver = CaptioningSolver(model, data, data, n_epochs=10, batch_size=100, update_rule='adam',
                              learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
                              pretrained_model=None, model_path='model/lstm_hard/', test_model='model/lstm_hard/model-40',
                              print_bleu=True, log_path='log/')

    test_data = load_coco_data(
        data_path='./data', split='test', if_train=False)
    solver.test(test_data, split='test')


if __name__ == "__main__":
    main()
