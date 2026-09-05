// ==========================================
// TRACENET AI - VEHICLE INTELLIGENCE
// API-connected dashboard
// ==========================================

document.addEventListener("DOMContentLoaded", () => {
    console.log("TRACENET AI System Started");

    const API_BASE = "http://127.0.0.1:8002";

    const totalVehicles = document.getElementById("totalVehicles");
    const trackedVehicles = document.getElementById("trackedVehicles");
    const databaseTable = document.getElementById("vehicleTableBody");
    const vehicleCountBadge = document.getElementById("vehicleCountBadge");
    const alertCount = document.getElementById("alertCount");

    const detectionStatus = document.getElementById("detectionStatus");
    const trackingStatus = document.getElementById("trackingStatus");
    const plateStatus = document.getElementById("plateStatus");

    const trafficChart = document.getElementById("trafficChart");
    const liveVideo = document.getElementById("liveVideo");

    const videoFileInput = document.getElementById("videoFileInput");
    const selectedFileName = document.getElementById("selectedFileName");
    const cameraIdInput = document.getElementById("cameraIdInput");
    const processVideoBtn = document.getElementById("processVideoBtn");
    const processingMessage = document.getElementById("processingMessage");

    // Alert elements
    const alertBanner = document.getElementById("alertBanner");
    const blacklistInput = document.getElementById("blacklistInput");
    const addBlacklistBtn = document.getElementById("addBlacklistBtn");
    const blacklistTags = document.getElementById("blacklistTags");
    const alertsContainer = document.getElementById("alertsContainer");
    const alertBadge = document.getElementById("alertBadge");
    const clearAlertsBtn = document.getElementById("clearAlertsBtn");

    let lastVehicles = [];
    let blacklistedPlates = new Set(
        JSON.parse(localStorage.getItem("blacklistedPlates") || '["SL688275", "DL4CAB1234"]')
    );
    let activeAlerts = [];

    // ==========================================
    // HELPERS
    // ==========================================

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function numberOrDash(value, digits = 2) {
        const n = Number(value);
        return Number.isFinite(n) ? n.toFixed(digits) : "-";
    }

    function getVehicleType(vehicle) {
        return (
            vehicle.class_name ||
            vehicle.vehicle ||
            vehicle.vehicle_class ||
            vehicle.class_name_label ||
            "Vehicle"
        );
    }

    function getTrackingId(vehicle) {
        return vehicle.tracking_id ?? vehicle.id ?? vehicle.track_id ?? "-";
    }

    function getPlate(vehicle) {
        const plate = (
            vehicle.plate_number ||
            vehicle.global_vehicle_candidate ||
            vehicle.plate ||
            vehicle.number_plate ||
            ""
        ).trim();
        return plate || "Not detected";
    }

    function getConfidence(vehicle) {
        return (
            vehicle.plate_confidence ??
            vehicle.confidence ??
            vehicle.vehicle_confidence ??
            null
        );
    }

    function isBlacklisted(plate) {
        if (!plate || plate === "Not detected") return false;
        const clean = plate.toUpperCase().replace(/[^A-Z0-9]/g, "");
        for (const item of blacklistedPlates) {
            const cleanItem = item.toUpperCase().replace(/[^A-Z0-9]/g, "");
            if (cleanItem && (clean.includes(cleanItem) || cleanItem.includes(clean))) {
                return true;
            }
        }
        return false;
    }

    // ==========================================
    // BLACKLIST MANAGEMENT
    // ==========================================

    function saveBlacklist() {
        localStorage.setItem("blacklistedPlates", JSON.stringify([...blacklistedPlates]));
        renderBlacklistTags();
    }

    function renderBlacklistTags() {
        if (!blacklistTags) return;
        blacklistTags.innerHTML = "";
        blacklistedPlates.forEach(plate => {
            const tag = document.createElement("span");
            tag.className = "blacklist-tag";
            tag.innerHTML = `
                <i class="fa-solid fa-ban"></i> ${escapeHtml(plate)}
                <button type="button" title="Remove" data-plate="${escapeHtml(plate)}">&times;</button>
            `;
            tag.querySelector("button").addEventListener("click", () => {
                blacklistedPlates.delete(plate);
                saveBlacklist();
                if (lastVehicles.length) updateDashboard(lastVehicles);
            });
            blacklistTags.appendChild(tag);
        });
    }

    if (addBlacklistBtn && blacklistInput) {
        addBlacklistBtn.addEventListener("click", () => {
            const val = blacklistInput.value.trim().toUpperCase();
            if (val) {
                blacklistedPlates.add(val);
                blacklistInput.value = "";
                saveBlacklist();
                if (lastVehicles.length) updateDashboard(lastVehicles);
            }
        });
        blacklistInput.addEventListener("keydown", e => {
            if (e.key === "Enter") {
                e.preventDefault();
                addBlacklistBtn.click();
            }
        });
    }

    if (clearAlertsBtn) {
        clearAlertsBtn.addEventListener("click", () => {
            activeAlerts = [];
            renderAlerts();
        });
    }

    function renderAlerts() {
        if (alertBadge) alertBadge.textContent = activeAlerts.length;
        if (alertCount) alertCount.textContent = activeAlerts.length;

        if (!alertsContainer) return;
        if (activeAlerts.length === 0) {
            alertsContainer.innerHTML = '<p class="no-alerts-msg">No alerts detected. System monitoring active.</p>';
            return;
        }

        alertsContainer.innerHTML = "";
        activeAlerts.forEach(alert => {
            const div = document.createElement("div");
            div.className = "alert-item";
            div.innerHTML = `
                <div>
                    <strong><i class="fa-solid fa-triangle-exclamation"></i> ${escapeHtml(alert.title)}:</strong>
                    ${escapeHtml(alert.message)}
                </div>
                <span style="font-size:11px;opacity:0.75;">${escapeHtml(alert.time)}</span>
            `;
            alertsContainer.appendChild(div);
        });
    }

    function addAlert(title, message) {
        const time = new Date().toLocaleTimeString();
        activeAlerts.unshift({ title, message, time });
        if (activeAlerts.length > 20) activeAlerts.pop();
        renderAlerts();
    }

    // ==========================================
    // DISPLAY VEHICLE DATA
    // ==========================================

    function updateDashboard(vehicles) {
        lastVehicles = Array.isArray(vehicles) ? vehicles : [];

        if (totalVehicles) {
            totalVehicles.textContent = lastVehicles.length;
        }

        if (vehicleCountBadge) {
            vehicleCountBadge.textContent = `${lastVehicles.length} vehicles`;
        }

        if (trackedVehicles) {
            const uniqueIds = new Set(
                lastVehicles
                    .map(getTrackingId)
                    .filter(id => id !== "-" && id !== null && id !== undefined)
                    .map(String)
            );
            trackedVehicles.textContent = uniqueIds.size || lastVehicles.length;
        }

        // Check for blacklisted vehicles and route anomalies
        activeAlerts = [];
        let blacklistedFound = 0;

        if (databaseTable) {
            databaseTable.innerHTML = "";

            if (lastVehicles.length === 0) {
                databaseTable.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align:center;color:#8297ad;padding:20px;">
                            Upload a video and click Process Video
                        </td>
                    </tr>
                `;
            } else {
                lastVehicles.forEach(vehicle => {
                    const plate = getPlate(vehicle);
                    const isFlagged = isBlacklisted(plate);
                    const trackId = getTrackingId(vehicle);
                    const type = getVehicleType(vehicle);
                    const confidence = getConfidence(vehicle);

                    if (isFlagged) {
                        blacklistedFound++;
                        addAlert("BLACKLIST MATCH", `Vehicle Track #${trackId} with plate [${plate}] matches security watchlist`);
                    }

                    const row = document.createElement("tr");
                    if (isFlagged) row.style.backgroundColor = "rgba(255, 77, 77, 0.15)";

                    row.innerHTML = `
                        <td><strong>#${escapeHtml(trackId)}</strong></td>
                        <td>${escapeHtml(plate)}</td>
                        <td><span style="text-transform:capitalize;">${escapeHtml(type)}</span></td>
                        <td>${confidence === null ? "-" : escapeHtml(numberOrDash(Number(confidence), 2))}</td>
                        <td>${isFlagged ? "<span class='alert-badge'><i class='fa-solid fa-triangle-exclamation'></i> BLACKLIST</span>" : "<span style='color:#36d97d;font-size:12px;'>Clear</span>"}</td>
                    `;

                    databaseTable.appendChild(row);
                });
            }
        }

        // Suspicious Route Anomaly Check: vehicles detected with unusual patterns
        if (lastVehicles.length > 0) {
            const trackGroups = {};
            lastVehicles.forEach(v => {
                const tid = getTrackingId(v);
                trackGroups[tid] = (trackGroups[tid] || 0) + 1;
            });

            // If a vehicle triggers multiple duplicate observations or abnormal duration
            for (const [tid, count] of Object.entries(trackGroups)) {
                if (count >= 3) {
                    addAlert("ROUTE ANOMALY", `Track #${tid} exhibited loitering / repetitive trajectory pattern`);
                }
            }
        }

        renderAlerts();

        if (detectionStatus) {
            detectionStatus.textContent = "ACTIVE";
        }

        if (trackingStatus) {
            trackingStatus.textContent = lastVehicles.length ? "ACTIVE" : "READY";
        }

        if (plateStatus) {
            const plates = lastVehicles.filter(v => getPlate(v) !== "Not detected");
            plateStatus.textContent = plates.length ? "ACTIVE" : "READY";
        }

        createTrafficChart(lastVehicles);
    }

    // ==========================================
    // PROCESS VIDEO THROUGH INTEGRATION API
    // ==========================================

    async function processVideo() {
        if (!videoFileInput?.files?.length) {
            setProcessingMessage("Please choose a video first.", true);
            return;
        }

        const file = videoFileInput.files[0];
        const cameraId = (cameraIdInput?.value || "CAM_01").trim() || "CAM_01";

        // Show and play selected video in the browser
        if (liveVideo) {
            const objectUrl = URL.createObjectURL(file);
            liveVideo.src = objectUrl;
            liveVideo.muted = true;
            liveVideo.load();
            liveVideo.play().catch(e => {
                console.log("Autoplay was prevented, click play on the video controls:", e);
            });
        }

        const formData = new FormData();
        formData.append("file", file, file.name);
        formData.append("camera_id", cameraId);

        processVideoBtn.disabled = true;
        processVideoBtn.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Processing...
        `;
        setProcessingMessage("Sending video to Tracking + ANPR AI pipelines...", false);

        try {
            const response = await fetch(`${API_BASE}/v1/process/video`, {
                method: "POST",
                body: formData
            });

            const result = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(
                    result.detail || `API request failed (${response.status})`
                );
            }

            const vehicles = Array.isArray(result.vehicles)
                ? result.vehicles
                : [];

            updateDashboard(vehicles);

            setProcessingMessage(
                `Processing complete — ${vehicles.length} vehicles tracked and analyzed (Camera: ${result.camera_id || cameraId}).`,
                false
            );

            console.log("Integration API result:", result);
        } catch (error) {
            console.error("Video processing failed:", error);
            setProcessingMessage(
                `Processing failed: ${error.message}. Ensure Integration (8002), Tracking (8001), and ANPR (8000) are active.`,
                true
            );
        } finally {
            processVideoBtn.disabled = false;
            processVideoBtn.innerHTML = `
                <i class="fa-solid fa-play"></i>
                Process Video
            `;
        }
    }

    function setProcessingMessage(message, isError) {
        if (!processingMessage) return;
        processingMessage.textContent = message;
        processingMessage.classList.toggle("error", Boolean(isError));
    }

    if (videoFileInput) {
        videoFileInput.addEventListener("change", () => {
            const file = videoFileInput.files?.[0];
            if (selectedFileName) {
                selectedFileName.textContent = file
                    ? file.name
                    : "Choose vehicle video";
            }
            // Auto preview on selection
            if (file && liveVideo) {
                liveVideo.src = URL.createObjectURL(file);
                liveVideo.muted = true;
                liveVideo.load();
                liveVideo.play().catch(() => {});
            }
        });
    }

    if (processVideoBtn) {
        processVideoBtn.addEventListener("click", processVideo);
    }

    // ==========================================
    // SERVICE HEALTH
    // ==========================================

    async function checkSystemHealth() {
        try {
            const response = await fetch(`${API_BASE}/health`);
            const health = await response.json();

            const integrationOk =
                health.status === "ok" &&
                health.tracking_ok === true &&
                health.anpr_ok === true;

            if (detectionStatus) {
                detectionStatus.textContent = integrationOk ? "ACTIVE" : "OFFLINE";
            }
            
            if (trackingStatus) {
                trackingStatus.textContent = integrationOk ? "ACTIVE" : "OFFLINE";
            }
            
            if (plateStatus) {
                plateStatus.textContent = integrationOk ? "READY" : "OFFLINE";
            }
        } catch (error) {
            console.warn("Could not reach Integration API:", error);

            if (detectionStatus) detectionStatus.textContent = "OFFLINE";
            if (trackingStatus) trackingStatus.textContent = "OFFLINE";
            if (plateStatus) plateStatus.textContent = "OFFLINE";
        }
    }

    // ==========================================
    // TRAFFIC ANALYTICS
    // ==========================================

    function createTrafficChart(vehicles) {
        if (!trafficChart) return;

        let cars = 0;
        let motorcycles = 0;
        let buses = 0;
        let trucks = 0;

        vehicles.forEach(vehicle => {
            const type = getVehicleType(vehicle).toLowerCase();

            if (type.includes("car")) {
                cars++;
            } else if (
                type.includes("motorcycle") ||
                type.includes("bike")
            ) {
                motorcycles++;
            } else if (type.includes("bus")) {
                buses++;
            } else if (type.includes("truck")) {
                trucks++;
            }
        });

        const chartData = [
            { label: "Cars", value: cars },
            { label: "Bikes", value: motorcycles },
            { label: "Buses", value: buses },
            { label: "Trucks", value: trucks }
        ];

        let maxValue = 1;
        chartData.forEach(item => {
            if (item.value > maxValue) maxValue = item.value;
        });

        trafficChart.innerHTML = "";

        chartData.forEach(item => {
            const chartItem = document.createElement("div");
            chartItem.className = "chart-item";

            const value = document.createElement("div");
            value.className = "chart-value";
            value.textContent = item.value;

            const barArea = document.createElement("div");
            barArea.className = "chart-bar-area";

            const bar = document.createElement("div");
            bar.className = "chart-bar";

            const height = (item.value / maxValue) * 160;
            bar.style.height = Math.max(height, 5) + "px";

            const label = document.createElement("div");
            label.className = "chart-label";
            label.textContent = item.label;

            barArea.appendChild(bar);
            chartItem.appendChild(value);
            chartItem.appendChild(barArea);
            chartItem.appendChild(label);
            trafficChart.appendChild(chartItem);
        });
    }

    // ==========================================
    // SETTINGS MODAL
    // ==========================================

    const settingsBtn = document.getElementById("settingsBtn");
    const settingsOverlay = document.getElementById("settingsOverlay");
    const closeSettings = document.getElementById("closeSettings");

    if (settingsBtn && settingsOverlay) {
        settingsBtn.addEventListener("click", event => {
            event.preventDefault();
            settingsOverlay.classList.add("show");
        });
    }

    if (closeSettings && settingsOverlay) {
        closeSettings.addEventListener("click", () => {
            settingsOverlay.classList.remove("show");
        });
    }

    if (settingsOverlay) {
        settingsOverlay.addEventListener("click", event => {
            if (event.target === settingsOverlay) {
                settingsOverlay.classList.remove("show");
            }
        });
    }

    // ==========================================
    // THEME SWITCHER
    // ==========================================

    const themeToggle = document.getElementById("themeToggle");
    const savedTheme = localStorage.getItem("theme");

    if (savedTheme === "light" && themeToggle) {
        document.body.classList.add("light-theme");
        themeToggle.checked = true;
    }

    if (themeToggle) {
        themeToggle.addEventListener("change", () => {
            if (themeToggle.checked) {
                document.body.classList.add("light-theme");
                localStorage.setItem("theme", "light");
            } else {
                document.body.classList.remove("light-theme");
                localStorage.setItem("theme", "dark");
            }
        });
    }

    // ==========================================
    // START APPLICATION
    // ==========================================

    renderBlacklistTags();
    renderAlerts();
    updateDashboard([]);
    checkSystemHealth();
    setInterval(checkSystemHealth, 5000);
});
