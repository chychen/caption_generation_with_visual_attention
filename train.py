from core.new_solver import CaptioningSolver
from core.new_model import CaptionGenerator
from core.utils import load_coco_data


def main():

    # load train dataset
    # data = load_coco_data(data_path='./data', split='train0', if_train=True)
    # word_to_idx = data['word_to_idx']
    # # load val dataset to print out bleu scores every epoch
    # val_data = load_coco_data(data_path='./data', split='val', if_train=False)
    # model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
    #                          dim_hidden=1024, n_time_step=16, prev2out=True,
    #                          ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    # solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=100, update_rule='adam',
    #                           learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
    #                           pretrained_model=None, model_path='model/lstm_hard/', test_model='model/lstm_hard/model-10',
    #                           print_bleu=True, log_path='log_hard/')
    # solver.train(chunk=0)

    # data = load_coco_data(data_path='./data', split='train1', if_train=True)
    # word_to_idx = data['word_to_idx']
    # val_data = load_coco_data(data_path='./data', split='val', if_train=False)
    # model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
    #                          dim_hidden=1024, n_time_step=16, prev2out=True,
    #                          ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    # solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=100, update_rule='adam',
    #                           learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
    #                           pretrained_model='model/lstm_hard/model-10', model_path='model/lstm_hard/', test_model='model/lstm_hard/model-20',
    #                           print_bleu=True, log_path='log_hard/')
    # solver.train(chunk=1)

    # data = load_coco_data(data_path='./data', split='train2', if_train=True)
    # word_to_idx = data['word_to_idx']
    # val_data = load_coco_data(data_path='./data', split='val', if_train=False)
    # model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
    #                          dim_hidden=1024, n_time_step=16, prev2out=True,
    #                          ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    # solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=100, update_rule='adam',
    #                           learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
    #                           pretrained_model='model/lstm_hard/model-20', model_path='model/lstm_hard/', test_model='model/lstm_hard/model-30',
    #                           print_bleu=True, log_path='log_hard/')
    # solver.train(chunk=2)

    data = load_coco_data(data_path='./data', split='train3', if_train=True)
    word_to_idx = data['word_to_idx']
    val_data = load_coco_data(data_path='./data', split='val', if_train=False)
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=1024, n_time_step=16, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=100, update_rule='adam',
                              learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
                              pretrained_model='model/lstm_hard/model-30', model_path='model/lstm_hard/', test_model='model/lstm_hard/model-40',
                              print_bleu=True, log_path='log_hard/')
    solver.train(chunk=3)

    # test_data = load_coco_data(data_path='./data', split='test', if_train=False)
    # solver.test(test_data, split='test')


if __name__ == "__main__":
    main()
