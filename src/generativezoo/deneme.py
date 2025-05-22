import requests
from simian.gui import Form, component, utils

if __name__ == "__main__":
    from simian.local import Uiformio
    Uiformio("deneme", window_title="Hierarchical VAE Model Eğitme")

# API Endpoint
HIERARCHICAL_VAE_TRAIN_API_URL = "http://localhost:8012/vanillavae_train"

# 📌 Kullanıcı arayüzünü oluştur
def gui_init(meta_data: dict) -> dict:
    form = Form()

    # 📌 Komponentleri liste formatında tanımla
    component_specs = [
        ["model_name", "Select", {"data": {"values": [{"label": "Hierarchical VAE", "value": "Hierarchical VAE"}]},
                                  "valueProperty": "value",
                                  "defaultValue": "Hierarchical VAE",
                                  "label": "Model Seç"}],

        ["dataset", "Select", {"data": {"values": [{"label": name.upper(), "value": name} for name in [
                                  "mnist", "cifar10", "cifar100", "places365", "dtd", "fashionmnist", 
                                  "chestmnist", "octmnist", "tissuemnist", "pneumoniamnist", "svhn", "tinyimagenet", "imagenet"]]},
                               "valueProperty": "value",
                               "defaultValue": "mnist",
                               "label": "Veri Seti Seç"}],

        ["learning_rate", "Number", {"default": 0.01, "label": "Öğrenme Oranı"}],
        ["n_epochs", "Number", {"default": 100, "label": "Epoch Sayısı"}],
        ["latent_dim", "Number", {"default": 512, "label": "Latent Boyutu"}],

        ["train_model", "Button", {"text": "Modeli Eğit", "event": "train_model", "label": "Modeli Eğit"}],
        ["training_status", "TextField", {"default": "Eğitim Durumu: Bekleniyor", "label": "Eğitim Durumu"}]
    ]

    # 📌 Komponentleri form'a ekle
    component_dict = utils.addComponentsFromTable(form, component_specs, ["key", "type", "options"])

    return {"form": form}

# 📌 Olayları yöneten fonksiyon
def gui_event(meta_data: dict, payload: dict) -> dict:
    Form.eventHandler(train_model=train_model)

    callback = utils.getEventFunction(meta_data, payload)
    return callback(meta_data, payload)

# 📌 Model eğitme fonksiyonu
def train_model(meta_data: dict, payload: dict) -> dict:
    model_name, _ = utils.getSubmissionData(payload, "model_name")
    dataset, _ = utils.getSubmissionData(payload, "dataset")
    learning_rate, _ = utils.getSubmissionData(payload, "learning_rate")
    n_epochs, _ = utils.getSubmissionData(payload, "n_epochs")
    latent_dim, _ = utils.getSubmissionData(payload, "latent_dim")

    data = {
        "model_name": model_name,
        "train": True,
        "dataset": dataset,
        "batch_size": 256,
        "n_epochs": int(n_epochs),
        "lr": float(learning_rate),
        "latent_dim": int(latent_dim),
        "sample_and_save_freq": 5,
        "num_workers": 0
    }

    try:
        res = requests.post(HIERARCHICAL_VAE_TRAIN_API_URL, json=data)
        message = res.json().get("message", "Model eğitilirken hata oluştu.")
        payload, _ = utils.setSubmissionData(payload, "training_status", message)
    except Exception as e:
        payload, _ = utils.setSubmissionData(payload, "training_status", f"API hatası: {str(e)}")

    return payload
