/**
 * charts.js — minimal, dependency-free SVG donut chart for the batch
 * analysis fraud/legitimate breakdown. No external charting library is
 * used, so this works fully offline.
 */
(function () {
  "use strict";

  function renderDonut(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;

    var fraud = Number(container.getAttribute("data-fraud") || 0);
    var legit = Number(container.getAttribute("data-legit") || 0);
    var total = fraud + legit;
    if (total <= 0) return;

    var radius = 54;
    var circumference = 2 * Math.PI * radius;
    var fraudFraction = fraud / total;
    var fraudLength = circumference * fraudFraction;

    var svgNS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 160 160");
    svg.setAttribute("width", "160");
    svg.setAttribute("height", "160");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Donut chart: " + fraud + " flagged fraud, " + legit + " legitimate");

    var bg = document.createElementNS(svgNS, "circle");
    bg.setAttribute("cx", "80");
    bg.setAttribute("cy", "80");
    bg.setAttribute("r", String(radius));
    bg.setAttribute("fill", "none");
    bg.setAttribute("stroke", "#e4f6ec");
    bg.setAttribute("stroke-width", "22");
    svg.appendChild(bg);

    var fg = document.createElementNS(svgNS, "circle");
    fg.setAttribute("cx", "80");
    fg.setAttribute("cy", "80");
    fg.setAttribute("r", String(radius));
    fg.setAttribute("fill", "none");
    fg.setAttribute("stroke", "#b3261e");
    fg.setAttribute("stroke-width", "22");
    fg.setAttribute("stroke-dasharray", fraudLength + " " + (circumference - fraudLength));
    fg.setAttribute("transform", "rotate(-90 80 80)");
    svg.appendChild(fg);

    var label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", "80");
    label.setAttribute("y", "86");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-size", "20");
    label.setAttribute("font-weight", "700");
    label.setAttribute("fill", "#0b1f3a");
    label.textContent = Math.round(fraudFraction * 100) + "%";
    svg.appendChild(label);

    var wrapper = document.createElement("div");
    wrapper.style.display = "flex";
    wrapper.style.alignItems = "center";
    wrapper.style.gap = "20px";
    wrapper.appendChild(svg);

    var legend = document.createElement("div");
    legend.innerHTML =
      '<div><span style="display:inline-block;width:10px;height:10px;background:#b3261e;border-radius:50%;margin-right:6px;"></span>Fraud: ' +
      fraud +
      "</div>" +
      '<div><span style="display:inline-block;width:10px;height:10px;background:#e4f6ec;border:1px solid #1c7c4d;border-radius:50%;margin-right:6px;"></span>Legitimate: ' +
      legit +
      "</div>";
    wrapper.appendChild(legend);

    container.innerHTML = "";
    container.appendChild(wrapper);
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderDonut("batchChart");
  });
})();
