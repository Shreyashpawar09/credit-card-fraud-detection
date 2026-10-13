/**
 * app.js — navigation toggle and client-side form validation.
 *
 * Server-side validation (src/validators.py) is the source of truth; this
 * script only gives users faster feedback and never bypasses the server
 * checks.
 */
(function () {
  "use strict";

  function setupNavToggle() {
    var toggle = document.getElementById("navToggle");
    var nav = document.getElementById("mainNav");
    if (!toggle || !nav) return;

    toggle.addEventListener("click", function () {
      var isOpen = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
    });
  }

  function isFiniteNumber(value) {
    if (value === "" || value === null || typeof value === "undefined") return false;
    var n = Number(value);
    return Number.isFinite(n);
  }

  function setupPredictFormValidation() {
    var form = document.getElementById("predictForm");
    if (!form) return;

    form.addEventListener("submit", function (event) {
      var inputs = form.querySelectorAll("input[type='number']");
      var allValid = true;

      inputs.forEach(function (input) {
        var value = input.value;
        var valid = isFiniteNumber(value);

        if (valid && (input.id === "Time" || input.id === "Amount") && Number(value) < 0) {
          valid = false;
        }

        input.classList.toggle("input-invalid", !valid);
        if (!valid) allValid = false;
      });

      if (!allValid) {
        event.preventDefault();
        var firstInvalid = form.querySelector(".input-invalid");
        if (firstInvalid) firstInvalid.focus();
      }
    });

    inputsClearOnEdit(form);

    var fillSampleBtn = document.getElementById("fillSampleBtn");
    if (fillSampleBtn) {
      fillSampleBtn.addEventListener("click", function () {
        fillSampleValues(form);
      });
    }
  }

  function inputsClearOnEdit(form) {
    form.querySelectorAll("input[type='number']").forEach(function (input) {
      input.addEventListener("input", function () {
        input.classList.remove("input-invalid");
      });
    });
  }

  /**
   * Fills the prediction form with illustrative example numbers so a user
   * can see how the UI behaves without needing a real transaction on hand.
   * These are arbitrary convenience values for UI exploration only — they
   * are not drawn from any real transaction and are not a dataset sample.
   */
  function fillSampleValues(form) {
    var timeInput = form.querySelector("#Time");
    var amountInput = form.querySelector("#Amount");
    if (timeInput) timeInput.value = Math.round(Math.random() * 100000);
    if (amountInput) amountInput.value = (Math.random() * 300).toFixed(2);

    for (var i = 1; i <= 28; i++) {
      var field = form.querySelector("#V" + i);
      if (field) {
        var sample = (Math.random() * 4 - 2).toFixed(4);
        field.value = sample;
      }
    }
    inputsClearOnEdit(form);
    form.querySelectorAll("input.input-invalid").forEach(function (el) {
      el.classList.remove("input-invalid");
    });
  }

  function setupBatchFormValidation() {
    var form = document.getElementById("batchForm");
    if (!form) return;

    form.addEventListener("submit", function (event) {
      var fileInput = form.querySelector("#csv_file");
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        event.preventDefault();
        alert("Please choose a CSV file to upload.");
        return;
      }
      var name = fileInput.files[0].name || "";
      if (!name.toLowerCase().endsWith(".csv")) {
        event.preventDefault();
        alert("Only .csv files are accepted.");
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupNavToggle();
    setupPredictFormValidation();
    setupBatchFormValidation();
  });
})();
