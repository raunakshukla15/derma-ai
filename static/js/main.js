const uploadBox = document.getElementById("uploadBox");
const imageInput = document.getElementById("imageInput");
const submitBtn = document.getElementById("submitBtn");
const report = document.getElementById("reportSection");

const uploadIcon = document.getElementById("uploadIcon");
const uploadLabel = document.getElementById("uploadLabel");

const aboutBtn = document.getElementById("aboutBtn");
const aboutPopup = document.getElementById("aboutPopup");
const closeAbout = document.getElementById("closeAbout");

const themeToggle = document.getElementById("themeToggle");

// Upload click
uploadBox.onclick = () => imageInput.click();

// Change icon after upload
imageInput.onchange = () => {
  if (imageInput.files.length > 0) {
    uploadIcon.textContent = "✅";
    uploadLabel.textContent = "IMAGE UPLOADED";
  }
};

// Submit
submitBtn.onclick = async () => {
  if (!imageInput.files[0]) {
    alert("Please upload an image first");
    return;
  }

  const fd = new FormData();
  fd.append("image", imageInput.files[0]);

  const res = await fetch("/predict", {
    method: "POST",
    body: fd
  });

  const data = await res.json();
  report.classList.remove("hidden");

  /* 🔧 FIX: remove (LOW CONFIDENCE) text */
  let cleanDiseaseName = data.message
    .replace(/possible/i, "")
    .replace(/\(.*?\)/g, "")
    .trim()
    .toUpperCase();

  document.getElementById("diseaseName").innerText = cleanDiseaseName;
  document.getElementById("desc").innerText = data.description;
  document.getElementById("sym").innerText = data.symptoms;
  document.getElementById("cure").innerText = data.cure;
  document.getElementById("home").innerText = data.agentic_doctor.home_care;
  document.getElementById("sevScore").innerText = data.severity_score;
  document.getElementById("sevLevel").innerText = data.severity_level;
  document.getElementById("warn").innerText = data.agentic_doctor.warning;
};

// Theme toggle
themeToggle.onclick = () => {
  document.body.classList.toggle("dark");
  themeToggle.textContent =
    document.body.classList.contains("dark") ? "☀️" : "🌙";
};

// About popup
aboutBtn.onclick = () => aboutPopup.classList.remove("hidden");
closeAbout.onclick = () => aboutPopup.classList.add("hidden");
