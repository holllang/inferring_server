import numpy as np
import json

def vectorize_sequences(sequences, dimension=40):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results

class InferModule:
    def __init__(self, model):
        self.model = model
        self.question_bias = []
        with open('./model_saved/base_info.json') as f:
            base_info = json.load(f)
            self.num_per_question = base_info["list"]
            self.num2hobby = base_info["hobby_enum"]

        for idx in range(len(self.num_per_question)):
            if idx == 0: 
                self.question_bias.append(0)
            else:
                self.question_bias.append(sum(self.num_per_question[:idx]))

    def start_inferring(self, infer_answer):
        X_infer = [(a+b-1) for a, b in zip(infer_answer, self.question_bias)]
        X_infer = vectorize_sequences([X_infer])
        predictions = self.model.predict(X_infer)
        
        for pred in predictions:
            ind = np.argpartition(pred, -3)[-3:]
            ind = ind[np.argsort(pred[ind])][::-1]
            hobby = []
            for i in ind:
                hobby.append(self.num2hobby[str(i)])
            return hobby