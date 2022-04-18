# !pip install pybind11
# !pip install fastwer

import fastwer
# Define reference text and output text
work_dir = "/home/chhabilal/Desktop/treeleaf/OCR_project"
for index in range(1,51):
    name = "actual_text/{index}.txt".format(index=index)
    path = os.path.join(work_dir, name)
    with io.open(path, mode="r", encoding="utf-8") as fd:
        reference_text = fd.read()
#         print(reference_text)
        
    name = 'tesseract_text/output{index}.txt'.format(index = index)
    path = os.path.join(work_dir, name)
    with io.open(path, mode = "r", encoding = "utf-8") as fd:
        predict_text = fd.read()
#         print(predict_text)

    print(fastwer.score_sent(predict_text, reference_text, char_level = True))
        
    