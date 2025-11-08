import torch
import numpy as np
import torch.nn.functional as F

from sklearn.metrics.pairwise import pairwise_distances

class NCMValidation(object):
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proto_list = []

    def _retrieval(self, val_loader):
        val_label_list = []
        val_embedding_list = []

        prototypes = torch.stack(self.proto_list, dim=0).cpu()
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        # Calculating validation image features
        with torch.no_grad():
            for _, batch in enumerate(val_loader):
                image, label = batch
                image = image.cuda()
                label = label.cuda()

                img_embedding = self.model(image)
                val_embedding_list.append(img_embedding.cpu())
                val_label_list.append(label.cpu())

        val_embedding_list = torch.cat(val_embedding_list, dim=0).cpu()
        val_embedding_list = F.normalize(val_embedding_list, p=2, dim=-1)
        val_label_list = torch.cat(val_label_list, dim=0).cpu()

        # Calculate cosine similarity (metric) of features and prototypes
        pairwise_distance = pairwise_distances(np.asarray(val_embedding_list), np.asarray(prototypes), metric='cosine')
        prediction_result = np.argmin(pairwise_distance, axis=1)

        val_label_list = np.asarray(val_label_list)
        top1_acc = np.sum(prediction_result == val_label_list) / float(len(val_label_list))

        return top1_acc

    def eval(self, train_loader, val_loader, num_train_cls, base_sess=True):
        self.calc_proto(train_loader, num_train_cls, base_sess)
        val_acc = self._retrieval(val_loader)
        val_acc = round(val_acc * 100, 2)
        return val_acc

    def calc_proto(self, train_loader, num_train_cls, base_sess=True):
        label_list = []
        embedding_list = []

        # Reset prototype list for each epoch of base session
        if base_sess:
            self.proto_list = []

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        # Generating prototypes using train image features
        with torch.no_grad():
            for _, (images, label) in enumerate(train_loader):
                images[0] = images[0].cuda()
                images[1] = images[1].cuda()
                label = label.cuda()
                
                img_embedding1 = self.model(images[0])                                   
                img_embedding2 = self.model(images[1])                                   
                img_embedding = (img_embedding1 + img_embedding2) / 2
                embedding_list.append(img_embedding.cpu())    
                label_list.append(label.cpu())
            
            embedding_list = torch.cat(embedding_list, dim=0)
            label_list = torch.cat(label_list, dim=0)

            # generate the average feature with all data
            start_cls_idx = len(self.proto_list)
            end_cls_idx = start_cls_idx + num_train_cls
            for index in range(start_cls_idx, end_cls_idx):
                class_index = (label_list == index).nonzero()
                embedding_this = embedding_list[class_index.squeeze(-1)]
                embedding_this = embedding_this.mean(0, keepdims=True).cuda()
                self.proto_list.append(embedding_this.view(-1))