# Model Latih tee-am-ai

Latih model tee-am-ai dengan gpt2

## Aktivasi virtual environment di cmd

1. Buat virtual environment

    <div style="border-radius: 10px; background: #f6f8fa; padding: 10px;">
        <pre><code>python -m venv .venv</code></pre>
    </div>

2. Aktivasi virtual environment

    ```bash
    .venv\Scripts\activate
    ```

3. Install semua library

    ```bash
    pip install -r requirements.txt
    ```

4. Running model

    ```bash
    python train.py
    ```

5. Testing model latih

    ```bash
    python test.py
    ```