### Usage

```Python
from model.infer import InferModule
from keras import models

model = models.load_model('./model_saved')
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    result = IM.start_inferring([1,2,4,1,2,3,2,2,1,2,3,1,1,2,3,2])
    print(result)
    
    # result : ['취미2', '취미4', '취미1']
```
