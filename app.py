import os
from flask import Flask, request, render_template_string
from PIL import Image
from transformers import pipeline
import base64
from io import BytesIO

app = Flask(__name__)

# Αρχικοποίηση του μοντέλου ταξινόμησης λουλουδιών
classifier = pipeline('image-classification', model='dima806/flower_groups_image_detection', framework='pt')

# Κύριο HTML template για το web interface
HTML = '''
<!doctype html>
<html lang="{{ lang }}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ 'Flower Classifier' if lang == 'en' else 'Ανιχνευτής Λουλουδιών' }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f8fafc; }
    .container { max-width: 500px; margin-top: 60px; background: #fff; border-radius: 12px; box-shadow: 0 2px 16px #0001; padding: 2rem; }
    .preview-img { max-width: 100%; border-radius: 8px; margin-bottom: 1rem; }
    .history-img { max-width: 60px; max-height: 60px; border-radius: 6px; object-fit: cover; margin-right: 8px; }
    .copy-btn { margin-left: 8px; }
    .dropzone { border: 2px dashed #aaa; border-radius: 8px; padding: 1.5rem; text-align: center; color: #888; margin-bottom: 1rem; cursor: pointer; transition: border-color 0.2s; }
    .dropzone.dragover { border-color: #007bff; color: #007bff; }
  </style>
</head>
<body>
  <div class="container">
    <h1 class="mb-4 text-center">🌸 {{ 'Flower Classifier' if lang == 'en' else 'Ανιχνευτής Λουλουδιών' }}</h1>
    <form method="post" enctype="multipart/form-data" class="mb-3" id="upload-form">
      <div id="dropzone" class="dropzone">{{ 'Drag & drop a flower photo here or click to select' if lang == 'en' else 'Σύρετε & αφήστε μια φωτογραφία λουλουδιού ή κάντε κλικ για επιλογή' }}</div>
      <input class="form-control mb-2 d-none" type="file" name="file" id="file-input" accept="image/*" required>
      <img id="preview" class="preview-img d-none" />
      <button class="btn btn-primary w-100" type="submit">{{ 'Predict Flower' if lang == 'en' else 'Πρόβλεψη Λουλουδιού' }}</button>
    </form>
    {% if pred %}
      <div class="alert alert-info text-center">
        <h4 class="mb-0">{{ 'Prediction:' if lang == 'en' else 'Πρόβλεψη:' }}</h4>
        <span class="fw-bold" id="main-pred">{{ pred }}</span>
        <button class="btn btn-sm btn-outline-secondary copy-btn" onclick="copyPred()">{{ 'Copy' if lang == 'en' else 'Αντιγραφή' }}</button>
        {% if top_preds %}
        <ul class="list-unstyled mt-2 mb-0">
          {% for label, score in top_preds %}
            <li>{{ label }} <span class="text-muted">({{ score }})</span></li>
          {% endfor %}
        </ul>
        {% endif %}
        <button class="btn btn-sm btn-success mt-2" onclick="downloadResult()">{{ 'Download Result' if lang == 'en' else 'Λήψη Αποτελέσματος' }}</button>
      </div>
    {% endif %}
    {% if history %}
      <hr>
      <h5>{{ 'Prediction History' if lang == 'en' else 'Ιστορικό Προβλέψεων' }}</h5>
      <ul class="list-group mb-0">
        {% for item in history %}
        <li class="list-group-item d-flex align-items-center">
          <img src="data:image/png;base64,{{ item.img_b64 }}" class="history-img me-2" />
          <span>{{ item.pred }}</span>
        </li>
        {% endfor %}
      </ul>
      <form method="post" action="/clear" class="mt-2 text-end">
        <button class="btn btn-sm btn-danger" type="submit">{{ 'Clear History' if lang == 'en' else 'Καθαρισμός Ιστορικού' }}</button>
      </form>
    {% endif %}
    <form method="get" class="mt-3 text-center">
      <button class="btn btn-link" name="lang" value="en" type="submit">English</button>|
      <button class="btn btn-link" name="lang" value="el" type="submit">Ελληνικά</button>
    </form>
  </div>
  <script>
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        const event = new Event('change');
        fileInput.dispatchEvent(event);
      }
    });
    fileInput.addEventListener('change', e => {
      const [file] = fileInput.files;
      if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.remove('d-none');
      } else {
        preview.classList.add('d-none');
      }
    });
    function copyPred() {
      const pred = document.getElementById('main-pred').innerText;
      navigator.clipboard.writeText(pred);
    }
    function downloadResult() {
      const pred = document.getElementById('main-pred').innerText;
      const text = `{{ 'Prediction' if lang == 'en' else 'Πρόβλεψη' }}: ${pred}\n`;
      const blob = new Blob([text], {type: 'text/plain'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'flower_prediction.txt';
      link.click();
    }
  </script>
</body>
</html>
'''

# Απλό ιστορικό στη μνήμη (όχι μόνιμο)
history = []

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    pred = None
    top_preds = None
    global history
    # Επιλογή γλώσσας (προεπιλογή: αγγλικά)
    lang = request.args.get('lang', 'en')
    if request.method == 'POST':
        lang = request.args.get('lang', request.form.get('lang', 'en'))
        f = request.files['file']
        img = Image.open(f.stream).convert('RGB')
        # Υπολογισμός πρόβλεψης για την εικόνα
        result = classifier(img, top_k=3)
        pred = result[0]['label'] if result else ('No prediction made' if lang == 'en' else 'Δεν έγινε πρόβλεψη')
        top_preds = [(r['label'], f"{r['score']*100:.1f}%") for r in result]
        buf = BytesIO()
        img.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        # Προσθήκη στο ιστορικό (μέχρι 5 εγγραφές)
        history.insert(0, {'img_b64': img_b64, 'pred': pred})
        history = history[:5]
    return render_template_string(HTML, pred=pred, top_preds=top_preds, history=history, lang=lang)

@app.route('/clear', methods=['POST'])
def clear_history():
    global history
    # Καθαρισμός ιστορικού
    history = []
    return upload_predict()

if __name__ == '__main__':
    # Εκκίνηση της εφαρμογής
    app.run(debug=True, host='0.0.0.0')
