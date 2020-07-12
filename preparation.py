import nltk
import random
import docx 
import summarizer
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

l = 1
k = 5

class FileLoader:

    @staticmethod
    def getTextFromDOCX(filepath):
      doc = docx.Document(filepath)
      fullText = []
      for para in doc.paragraphs:
          fullText.append(para.text)
      return '\n'.join(fullText)

    @staticmethod
    def getTextFromTXT(filepath):
      fullText = []
      with open(filepath, 'rb') as file:
        for line in file.readlines():
          fullText.append(line.decode('UTF-8'))
          #fullText.append(line)
      return '\n'.join(fullText)

class FilePreprocessor:

    def __init__(self, filepath, filetype):
      self.text = [FileLoader.getTextFromDOCX(filepath) if filetype == 'Word file (*.docx)'\
                  else FileLoader.getTextFromTXT(filepath),]

    def splitTextRandomly(self, l):
      text = [part.replace('\n', ' ') for part in self.text]
      text = [' '.join(text)]
      indices = [i for i, x in enumerate(text[0]) if x == "."]
      if len(indices) < l:
        return self.splitTextEvenly(l)
      else:
        dot_idx = random.sample(indices, l)
        dot_idx = sorted(dot_idx)[:-1]
        dot_idx.append(None)
        dot_idx = [0] + dot_idx
        print(dot_idx)
        text = [text[0][dot_idx[i] + 1 if text[0][dot_idx[i]] == '.' else 0:dot_idx[i+1]]\
                    for i in range(len(dot_idx) - 1)\
                    if dot_idx[i] + 1 < len(text[0])]

      return text

    def splitTextEvenly(self, l):
      text = [part.replace('\n', ' ') for part in self.text]
      text = [' '.join(text)]
      indices = [i for i, x in enumerate(text[0]) if x == "."]
      if len(indices) < l:
          return ['Деление текста не удалось.']
      lengthOfPart = len(text[0])/(l + 1)
      dot_idx = []
      part_num = 1
      for idx in indices:
        if idx > part_num * lengthOfPart:
          dot_idx.append(idx)
          part_num += 1
      dot_idx.append(None)
      dot_idx = [0] + dot_idx
      #text = [text[0][:dot_idx+1], text[0][dot_idx+1:]]
      text = [text[0][dot_idx[i] + 1 if text[0][dot_idx[i]] == '.' else 0:dot_idx[i+1]]\
                   for i in range(len(dot_idx) - 1)\
                   if dot_idx[i] + 1 < len(text[0])]

      return text


    def joinText(self):
      text = [part.replace('\n', ' ') for part in self.text]
      text = [' '.join(text)]

      return text

