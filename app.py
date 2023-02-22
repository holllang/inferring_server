from flask import Flask, jsonify, request

from model.infer import InferModule
from keras import models

model = models.load_model('./model/model_saved.h5')
IM = InferModule(model)

app = Flask(__name__)

@app.route('/test-responses', methods=['POST'])
def inferHobbiesAndType():
    testResponses: list = request.get_json()
    # 질문 번호순서대로 정렬
    testResponses.sort(key=lambda x: x['questionNumber'])
    #리스트로 변환
    inferringInputData = []
    for testResponse in testResponses:
        inferringInputData.append(testResponse['answerNumber'])
    #추론 딥러닝 모델에 추론 요청
    inferringResponse = IM.start_inferring(inferringInputData)

    return jsonify(inferringResponse), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
