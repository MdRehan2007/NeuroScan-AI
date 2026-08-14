document.addEventListener("DOMContentLoaded", () => {
  const uploadBox = document.getElementById("upload-box");
  const fileInput = document.getElementById("file-input");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview-img");
  const fileNameEl = document.getElementById("file-name");
  const predictBtn = document.getElementById("predict-btn");
  const resetBtn = document.getElementById("reset-btn");
  const resultCard = document.getElementById("result-card");
  const resultClass = document.getElementById("result-class");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceFill = document.getElementById("confidence-fill");
  const errorBox = document.getElementById("error-box");
  const spinner = document.getElementById("spinner");

  let selectedFile = null;

  function resetResult() {
    resultCard.classList.remove("visible");
    errorBox.classList.remove("visible");
    errorBox.textContent = "";
    confidenceFill.style.width = "0%";
  }

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add("visible");
  }

  function handleFile(file) {
    if (!file) return;

    const allowed = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowed.includes(file.type)) {
      showError("Please select a JPG, JPEG, or PNG image.");
      return;
    }

    selectedFile = file;
    resetResult();

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewWrap.style.display = "block";
    };
    reader.readAsDataURL(file);

    fileNameEl.textContent = file.name;
    predictBtn.disabled = false;
  }

  uploadBox.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    uploadBox.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadBox.classList.remove("dragover");
    });
  });

  uploadBox.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      handleFile(file);
    }
  });

  resetBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewWrap.style.display = "none";
    fileNameEl.textContent = "";
    predictBtn.disabled = true;
    resetResult();
  });

  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Please choose an MRI image first.");
      return;
    }

    resetResult();
    spinner.classList.add("visible");
    predictBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || "Something went wrong while predicting.");
        return;
      }

      resultClass.textContent = data.prediction;
      confidenceValue.textContent = data.confidence.toFixed(2) + "%";
      resultCard.classList.add("visible");
      requestAnimationFrame(() => {
        confidenceFill.style.width = data.confidence + "%";
      });
    } catch (err) {
      showError("Could not reach the server. Please check that it is running.");
    } finally {
      spinner.classList.remove("visible");
      predictBtn.disabled = false;
    }
  });
});
