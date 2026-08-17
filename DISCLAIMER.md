# Disclaimer

Please read this before training on, or uploading, any dataset or model with LoRA Fine-tune Studio.

## You run this entirely on your own machine, with your own credentials

LoRA Fine-tune Studio is a local, single-user application. There is no hosted version, no backend server operated by the author, and no account system. Training runs on your own CUDA GPU, using whichever datasets and base models you provide. Your Hugging Face token (`HF_TOKEN`), when set, is read only from your environment or your local `.streamlit/secrets.toml` and is never written into saved training configuration, worker logs, or this repository — see [SECURITY.md](SECURITY.md).

## You are responsible for the data and models you process

**You, and only you, are responsible for:**

- Having the rights or permission to use every dataset and base model you load, whether from the Hugging Face Hub or a local upload, and complying with its license, terms of use, and any privacy or contractual obligations that apply to it.
- Deciding whether an adapter may be uploaded to the Hugging Face Hub. LoRA Fine-tune Studio only uploads when you explicitly select **Push adapter to Hugging Face Hub** — nothing is uploaded automatically.
- Reviewing the source, license, and integrity of any downloaded model, dataset, or tokenizer before training on it. The application keeps `trust_remote_code=False` and requires safetensors as a default safety boundary, but this does not make arbitrary third-party content trustworthy.
- Not training secrets, personal information, or restricted data into model weights without an approved governance and retention process of your own.
- Any Hugging Face Hub costs, rate limits, or terms associated with your account and token.
- Verifying that a trained adapter behaves as intended before relying on it, deploying it, or sharing it further.

## No warranty, no liability

This software is provided "as is," without warranty of any kind, as stated in the [MIT License](LICENSE). The author is not liable for any damage, data loss, GPU hardware issues, unintended disclosure, Hub costs, or other consequences arising from your use of this tool. Use it at your own risk.

## No financial support wanted

This project is free, open-source, and does not want or accept donations, sponsorships, or any other form of financial contribution — see [SUPPORT.md](SUPPORT.md).
