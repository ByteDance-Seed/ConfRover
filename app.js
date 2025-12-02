// Copyright 2025 Bytedance Ltd. and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const tasks = {
  "ATLAS 100ns Simulation": [
    { name: "6h86_A", folderUrl: "assets/atlas_100ns/6h86_A", desc: "Synaptonemal complex central element protein 3" },
    { name: "7asg_A", folderUrl: "assets/atlas_100ns/7asg_A", desc: "Transforming growth factor-beta-induced protein TGFBIp mutant" },
    { name: "6qj0_A", folderUrl: "assets/atlas_100ns/6qj0_A", desc: "Condensin Smc2 ATPase head" },
    { name: "6in7_A", folderUrl: "assets/atlas_100ns/6in7_A", desc: "Anti-sigma factor MucA" },
    { name: "7c45_A", folderUrl: "assets/atlas_100ns/7c45_A", desc: "Trypanosoma brucei RNase D" },
    { name: "6l4l_A", folderUrl: "assets/atlas_100ns/6l4l_A", desc: "Diaminopimelate epimerase" },
    { name: "6xds_A", folderUrl: "assets/atlas_100ns/6xds_A", desc: "MBP-TREM2 Ig domain fusion" },
    { name: "7bwf_B", folderUrl: "assets/atlas_100ns/7bwf_B", desc: "Antitoxin" },
  ],
};

let activeTask = Object.keys(tasks)[0];
let activeCaseIndex = 0;
let viewer;
let statusEl;
// Create tabs from tasks definition
function initTabs() {
  const tabs = document.getElementById("task-tabs");
  tabs.innerHTML = "";
  Object.keys(tasks).forEach(taskName => {
    const b = document.createElement("button");
    b.className = "tab" + (taskName === activeTask ? " active" : "");
    b.textContent = taskName;
    b.onclick = () => {
      activeTask = taskName;
      activeCaseIndex = 0;
      if (viewer && viewer.plugin) {
        viewer.plugin.clear();
      }
      initTabs();
      renderCases();
    };
    tabs.appendChild(b);
  });
}

function renderCases() {
  const list = document.getElementById("case-list");
  list.innerHTML = "";
  tasks[activeTask].forEach((c, i) => {
    const item = document.createElement("div");
    item.className = "case" + (i === activeCaseIndex ? " active" : "");
    const left = document.createElement("div");
    left.className = "case-name";
    const caseName = c.name.replaceAll('_', '-').toUpperCase();
    left.textContent = caseName;
    const right = document.createElement("div");
    right.className = "case-meta";
    if (c.desc) {
      right.textContent = c.desc;
    } else if (c.folderUrl) {
      right.textContent = `${c.name}.pdb`;
    } else {
      right.textContent = c.pdbUrl.split("/").pop();
    }
    item.appendChild(left);
    item.appendChild(right);
    item.onclick = () => {
      activeCaseIndex = i;
      renderCases();
      loadSelectedCase();
    };
    list.appendChild(item);
  });
}

function initViewer() {
  if (typeof molstar === 'undefined' || !molstar.Viewer) {
    setStatus('Mol* not available. Check network/CDN.');
    return;
  }
  return molstar.Viewer.create("molstar-viewer", {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowRemoteState: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    layoutShowRightPanel: false,
    collapseLeftPanel: true,
    collapseRightPanel: true,
    // simple view
    viewportShowControls: false,
    viewportShowSettings: false,           // hide gear icon
    viewportShowExpand: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: true,
    viewportShowTrajectoryControls: false,  // show timeline & play controls
    pdbProvider: 'rcsb',
    emdbProvider: 'rcsb'
  }).then(v => {
    viewer = v;
    window.viewer = v;
    setStatus('Mol* loaded.');
  });
}

async function loadSelectedCase() {
  const c = tasks[activeTask][activeCaseIndex];
  if (!viewer) await initViewer();
  const caseName = c.name.replaceAll('_', '-').toUpperCase();
  await viewer.plugin.clear();
  setStatus(`Loading ${caseName} ...`);
  try {
    // parse input: if c contains folderUrl, load the trajectory {c.name}.pdb and {c.name}.xtc files under the folder
    if (c.folderUrl) {
      console.log(`Loading trajectory ${c.folderUrl}/${c.name}.pdb and ${c.folderUrl}/${c.name}.xtc`);
      const xtcPath = `${c.folderUrl}/${c.name}_smoothed.xtc`;
      const pdbPath = `${c.folderUrl}/${c.name}.pdb`;
      await viewer.loadTrajectory({
        model: {
          kind: 'model-url',
          url: pdbPath,
          format: 'pdb',
          isBinary: false
        },
        coordinates: {
          kind: 'coordinates-url',
          url: xtcPath,
          format: 'xtc',
          isBinary: true
        },

        preset: 'default'
      });
      setStatus(`Loaded ${caseName} smoothed trajectory.`);
      await applyColorTheme('secondary-structure')
      // await applyColorTheme('illustrative')
      autoClickPlayOnce();
    }
    else {
      await viewer.loadStructureFromUrl(c.pdbUrl, "pdb");
      setStatus(`Loaded ${c.name}.`);
    }
    return;
  } catch (e) {
    setStatus(`Failed to load ${c.name}: ${e && e.message ? e.message : e}`);
  }
}

async function applyColorTheme(colorName) {
  const plugin = viewer.plugin;

  await plugin.dataTransaction(async () => {
    for (const s of plugin.managers.structure.hierarchy.current.structures) {
      await plugin.managers.structure.component.updateRepresentationsTheme(
        s.components,
        { color: colorName }    // e.g. 'sequence-id', 'chain-id', 'illustrative', 'uniform', 'uncertainty'
      );
    }
  });
}

function autoClickPlayOnce() {
  const container = document.getElementById('molstar-viewer');

  // 1) Open the controls panel (upper-left toggle)
  //    You may need to tweak this selector based on your DOM.
  const controlsToggle = container.querySelector('[title="Select Animation"]');
  if (controlsToggle) {
    controlsToggle.click();
  }

  // 2) Wait for the animation/trajectory controls to render
  const observer = new MutationObserver(() => {
    // Try to find the play button in trajectory/animation controls
    const playBtn = container.querySelector(
      '.msp-animation-viewport-controls-select .msp-flex-row > button'
    );
    if (playBtn) {
      // 3) Click it once to start the animation
      playBtn.click();

      observer.disconnect();
    }
  });

  observer.observe(container, { childList: true, subtree: true });
}


function init() {
  statusEl = document.getElementById('molstar-status');
  initTabs();
  renderCases();
  setStatus('Click on any case to load it.');
  setupSectionReveal();
}

function setStatus(msg) {
  if (!statusEl) return;
  statusEl.textContent = msg;
}

function setupSectionReveal() {
  const sections = document.querySelectorAll('.section');
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('is-visible');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });
  sections.forEach(s => obs.observe(s));
}

window.addEventListener("DOMContentLoaded", init);
