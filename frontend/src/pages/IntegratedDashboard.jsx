import React, { useEffect, useMemo, useRef, useState } from "react";
import "./IntegratedDashboard.css";

const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const RAW_API_BASE =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  DEFAULT_API_BASE;
const API_BASE = RAW_API_BASE.replace(/\/$/, "");

const PLAY_INTERVAL_MS = 1000;
const PAGE_SIZE = 5;
const CODE_VERSION = "integrated_v5_front_ESTIMATION_ONLY";

const ABNORMAL_ALLOWED_LABELS = [
  "기타",
  "수면",
  "식사",
  "외출",
  "주의",
  "위험",
];

const ABNORMAL_SCORE_MAP = {
  기타: 18,
  수면: 22,
  식사: 30,
  외출: 38,
  주의: 65,
  위험: 92,
};

function safeNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function clamp(value, min = 0, max = 100) {
  return Math.max(min, Math.min(max, value));
}

function splitCsvLine(line) {
  const result = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  result.push(current.trim());
  return result;
}

function normalizeColumnKey(value) {
  return String(value || "")
    .replace("\ufeff", "")
    .trim()
    .toLowerCase()
    .replace(/[\s_\-]/g, "");
}

function findEstimationIndex(headers) {
  const strictCandidates = ["estimation"];
  const fallbackCandidates = [
    "상태",
    "라벨",
    "행동상태",
    "이상행동",
    "분류",
    "판정",
    "결과",
  ];

  for (const candidate of strictCandidates) {
    const idx = headers.findIndex(
      (header) => normalizeColumnKey(header) === candidate,
    );
    if (idx >= 0) return idx;
  }

  for (const candidate of fallbackCandidates) {
    const idx = headers.findIndex(
      (header) => normalizeColumnKey(header) === normalizeColumnKey(candidate),
    );
    if (idx >= 0) return idx;
  }

  return -1;
}

function extractEstimationLabelsFromCsvText(text) {
  const lines = String(text || "")
    .replace(/^\ufeff/, "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (lines.length <= 1) {
    return {
      rowCount: 0,
      estimationColumn: null,
      labels: [],
      error: "CSV 행이 부족합니다.",
    };
  }

  const headers = splitCsvLine(lines[0]);
  const estimationIndex = findEstimationIndex(headers);

  if (estimationIndex < 0) {
    return {
      rowCount: lines.length - 1,
      estimationColumn: null,
      labels: [],
      error: "Estimation 컬럼을 찾지 못했습니다.",
    };
  }

  const labels = [];

  for (let i = 1; i < lines.length; i += 1) {
    const cells = splitCsvLine(lines[i]);
    const raw = cells[estimationIndex];
    labels.push(normalizeAbnormalLabel(raw));
  }

  return {
    rowCount: labels.length,
    estimationColumn: headers[estimationIndex],
    labels,
    error: null,
  };
}

function patchResultWithClientEstimation(item, labels) {
  if (!item || !Array.isArray(labels) || labels.length === 0) return item;

  const index = Math.max(0, safeNumber(item.step, 1) - 1);
  const rawLabel = labels[index];

  if (!rawLabel) return item;

  const state = normalizeAbnormalLabel(rawLabel);
  const level = abnormalLevelFromLabel(state);
  const riskScore = abnormalScoreFromLabel(state);
  const abnormal = item.abnormal || {};

  const patchedAbnormal = {
    ...abnormal,
    module: "abnormal",
    title: "이상행동",
    state,
    level,
    risk_score: riskScore,
    behavior: state,
    abnormal_type: state,
    reason:
      abnormal.reason ||
      `프론트에서 CSV Estimation ${index + 1}행 값을 직접 표시했습니다.`,
    detail:
      abnormal.detail ||
      `프론트에서 CSV Estimation ${index + 1}행 값을 직접 표시했습니다.`,
    features: {
      ...(abnormal.features || {}),
      frontend_code_version: CODE_VERSION,
      frontend_state_source: "client_csv_estimation_column",
      frontend_row_index: index,
      frontend_estimation: state,
    },
  };

  const fallDanger = item.fall?.level === "danger";
  const overallLevel = fallDanger ? "danger" : level;
  const overallLabel = fallDanger
    ? "위험"
    : state === "위험"
      ? "위험"
      : state === "주의"
        ? "주의"
        : "정상";

  return {
    ...item,
    abnormal: patchedAbnormal,
    overall: {
      ...(item.overall || {}),
      level: overallLevel,
      label: overallLabel,
      risk_score: Math.max(safeNumber(item.fall?.risk_score), riskScore),
      message: `이상행동은 CSV Estimation ${index + 1}행(${state}) 기준으로 표시합니다.`,
    },
  };
}

function getLockedAbnormalRows(status) {
  return safeNumber(
    status?.profile?.abnormal_rows || status?.abnormal_row_count,
    0,
  );
}

function getDisplayTotalSeconds(status) {
  const abnormalRows = getLockedAbnormalRows(status);

  if (abnormalRows > 0) {
    return abnormalRows;
  }

  return safeNumber(status?.total_seconds, 0);
}

function getDisplayCurrentSeconds(status) {
  const abnormalRows = getLockedAbnormalRows(status);

  if (abnormalRows > 0) {
    return Math.min(safeNumber(status?.current_step, 0), abnormalRows);
  }

  return safeNumber(status?.current_seconds, 0);
}

function formatTime(value) {
  if (!value) return "-";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatCm(value) {
  const cm = safeNumber(value, 0) * 100;
  return `${cm.toFixed(1)} cm`;
}

function formatSpeed(value) {
  return `${safeNumber(value, 0).toFixed(2)} m/s`;
}

function formatPercent(value) {
  const num = safeNumber(value, 0);

  if (num <= 1) {
    return `${Math.round(num * 100)}%`;
  }

  return `${Math.round(num)}%`;
}

function normalizeAbnormalLabel(value) {
  const text = String(value || "")
    .replace("\ufeff", "")
    .trim();

  const lower = text.toLowerCase();

  if (ABNORMAL_ALLOWED_LABELS.includes(text)) {
    return text;
  }

  if (
    !text ||
    text === "-" ||
    lower === "nan" ||
    lower === "none" ||
    lower === "null"
  ) {
    return "기타";
  }

  const compact = lower
    .replace(/\s/g, "")
    .replace(/_/g, "")
    .replace(/-/g, "")
    .replace(/\//g, "");

  const numericMap = {
    0: "기타",
    1: "수면",
    2: "식사",
    3: "외출",
    4: "주의",
    5: "위험",
  };

  if (Object.prototype.hasOwnProperty.call(numericMap, compact)) {
    return numericMap[compact];
  }

  const hasAny = (keywords) =>
    keywords.some((keyword) => {
      const key = String(keyword)
        .toLowerCase()
        .replace(/\s/g, "")
        .replace(/_/g, "")
        .replace(/-/g, "")
        .replace(/\//g, "");

      return compact.includes(key);
    });

  if (
    hasAny(["위험", "danger", "emergency", "critical", "highrisk", "riskhigh"])
  ) {
    return "위험";
  }

  if (
    hasAny([
      "주의",
      "warning",
      "caution",
      "abnormal",
      "care",
      "watch",
      "wandering",
      "inactive",
      "inactivity",
      "noactivity",
      "longstay",
    ])
  ) {
    return "주의";
  }

  if (
    hasAny([
      "외출",
      "outing",
      "outside",
      "outdoor",
      "goout",
      "leave",
      "leaving",
      "out",
    ])
  ) {
    return "외출";
  }

  if (
    hasAny([
      "식사",
      "meal",
      "eat",
      "eating",
      "food",
      "breakfast",
      "lunch",
      "dinner",
    ])
  ) {
    return "식사";
  }

  if (
    hasAny([
      "수면",
      "sleep",
      "sleeping",
      "rest",
      "resting",
      "bed",
      "lying",
      "lie",
    ])
  ) {
    return "수면";
  }

  return "기타";
}

function pickAbnormalLabel(data) {
  if (!data) return "대기";

  if (data.level === "idle" || data.state === "대기") {
    return "대기";
  }

  // 중요:
  // 백엔드가 계산해서 내려준 최종 state를 그대로 표시한다.
  // 이전 코드는 features.csv_estimation / model_state를 state보다 먼저 봐서
  // 백엔드가 기타로 고쳐도 화면에서는 다시 주의로 덮어쓰는 문제가 있었다.
  const directState = data?.state || data?.features?.final_state;

  if (directState !== null && directState !== undefined) {
    const text = String(directState).replace("\ufeff", "").trim();

    if (text && text !== "-") {
      return normalizeAbnormalLabel(text);
    }
  }

  const fallbackCandidates = [
    data?.abnormal_type,
    data?.behavior,
    data?.features?.rule_state,
  ];

  for (const value of fallbackCandidates) {
    if (value === null || value === undefined) continue;

    const text = String(value).replace("\ufeff", "").trim();

    if (!text || text === "-") continue;

    return normalizeAbnormalLabel(text);
  }

  return "기타";
}

function abnormalLevelFromLabel(label) {
  const state = normalizeAbnormalLabel(label);

  if (state === "위험") return "danger";
  if (state === "주의") return "warning";

  return "normal";
}

function abnormalScoreFromLabel(label) {
  const state = normalizeAbnormalLabel(label);
  return ABNORMAL_SCORE_MAP[state] ?? 18;
}

function getAbnormalDisplayData(data) {
  if (!data) {
    return {
      state: "대기",
      level: "idle",
      risk_score: 0,
      reason: "CSV 업로드 후 결과가 표시됩니다.",
      abnormal_type: "-",
      behavior: "-",
      guardian_alert: false,
      heart_rate: null,
      respiratory_rate: null,
      temperature: null,
      stress_score: null,
      features: {},
    };
  }

  const state = pickAbnormalLabel(data);

  if (state === "대기") {
    return {
      ...data,
      state: "대기",
      level: "idle",
      risk_score: 0,
      abnormal_type: "-",
      behavior: "-",
      features: data.features || {},
    };
  }

  const riskScore = abnormalScoreFromLabel(state);

  return {
    ...data,
    state,
    level: abnormalLevelFromLabel(state),
    risk_score: riskScore,
    behavior: state,
    abnormal_type: state,
    features: data.features || {},
  };
}

function levelLabel(level) {
  if (level === "danger") return "위험";
  if (level === "warning") return "주의";
  if (level === "normal") return "정상";
  return "대기";
}

function StatusBadge({ level, text }) {
  return (
    <span className={`integrated-badge ${level || "idle"}`}>
      {text || levelLabel(level)}
    </span>
  );
}

function getVitalDisplay(data) {
  const level = data?.level || "idle";
  const waiting = !data || level === "idle" || data?.state === "대기";
  const anomaly =
    Boolean(data?.is_anomaly) || level === "danger" || level === "warning";

  if (waiting) {
    return {
      level: "idle",
      label: "대기",
      badgeText: "대기",
      detail: "호흡 CSV 업로드 후 결과가 표시됩니다.",
    };
  }

  return {
    level: anomaly ? "warning" : "normal",
    label: anomaly ? "이상 있음" : "이상 없음",
    badgeText: anomaly ? "이상" : "정상",
    detail:
      data?.reason ||
      (anomaly
        ? "호흡 신호에서 이상 가능성이 감지되었습니다."
        : "호흡 신호가 정상 범위입니다."),
  };
}

function RiskBar({ label, value, level, valueText, percent }) {
  const barValue = clamp(safeNumber(percent ?? value));

  return (
    <div className="integrated-risk-row">
      <div className="integrated-risk-top">
        <span>{label}</span>
        <strong>{valueText || `${Math.round(safeNumber(value))}점`}</strong>
      </div>

      <div className="integrated-risk-track">
        <div
          className={`integrated-risk-fill ${level || "idle"}`}
          style={{ width: `${barValue}%` }}
        />
      </div>
    </div>
  );
}

function UploadBox({
  title,
  desc,
  file,
  onChange,
  compact = false,
  note = null,
}) {
  return (
    <label
      className={`integrated-upload-box ${file ? "selected" : ""} ${
        compact ? "compact" : ""
      }`}
    >
      <input
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={(event) => onChange(event.target.files?.[0] || null)}
      />
      <span>{title}</span>
      <strong>{file ? file.name : "파일 선택"}</strong>
      <p>{desc}</p>
      {note && <em>{note}</em>}
    </label>
  );
}

function findBestModule(history, moduleName) {
  if (!history || history.length === 0) return null;

  return history.reduce((best, item) => {
    const bestScore = safeNumber(best?.[moduleName]?.risk_score, -1);
    const score = safeNumber(item?.[moduleName]?.risk_score, -1);

    return score > bestScore ? item : best;
  }, history[0]);
}

function findBestFall(history) {
  const dangerFalls = history.filter((item) => item?.fall?.level === "danger");

  if (dangerFalls.length > 0) {
    return dangerFalls.reduce((best, item) => {
      const bestScore = safeNumber(best?.fall?.risk_score);
      const score = safeNumber(item?.fall?.risk_score);
      return score > bestScore ? item : best;
    }, dangerFalls[0]);
  }

  return findBestModule(history, "fall");
}

function getBehaviorLabel(item) {
  const abnormal = item?.abnormal;

  if (!abnormal) {
    return "대기";
  }

  return pickAbnormalLabel(abnormal);
}

function getBehaviorLevel(item) {
  const label = getBehaviorLabel(item);

  if (label === "대기") return "idle";
  if (label === "위험") return "danger";
  if (label === "주의") return "warning";

  return "normal";
}

function getBehaviorClass(label) {
  const state = normalizeAbnormalLabel(label);

  if (state === "위험") return "danger";
  if (state === "주의") return "warning";
  if (state === "수면") return "sleep";
  if (state === "식사") return "meal";
  if (state === "외출") return "out";
  if (state === "기타") return "etc";

  return "idle";
}

function buildBehaviorSegments(history, status) {
  if (!history || history.length === 0) return [];

  const ordered = [...history].reverse();

  const defaultWindowSeconds =
    safeNumber(status?.step_seconds) ||
    safeNumber(ordered[0]?.window?.step_seconds) ||
    safeNumber(ordered[0]?.window?.window_seconds) ||
    1;

  const segments = [];

  ordered.forEach((item) => {
    const second = safeNumber(item.second);
    const windowSeconds =
      safeNumber(item?.window?.step_seconds) ||
      safeNumber(item?.window?.window_seconds) ||
      defaultWindowSeconds;

    const label = getBehaviorLabel(item);
    const level = getBehaviorLevel(item);
    const labelClass = getBehaviorClass(label);
    const guardianAlert =
      label === "위험" && Boolean(item?.abnormal?.guardian_alert);
    const riskScore = label === "대기" ? 0 : abnormalScoreFromLabel(label);
    const reason = item?.abnormal?.reason || item?.abnormal?.detail || "-";

    const current = {
      label,
      level,
      labelClass,
      guardianAlert,
      riskScore,
      reason,
      start: second,
      end: second + windowSeconds,
    };

    const prev = segments[segments.length - 1];

    if (
      prev &&
      prev.label === current.label &&
      prev.level === current.level &&
      prev.guardianAlert === current.guardianAlert
    ) {
      prev.end = current.end;
      prev.riskScore = Math.max(prev.riskScore, current.riskScore);
    } else {
      segments.push(current);
    }
  });

  return segments;
}

function ModuleCard({ title, data }) {
  const abnormalData =
    title === "이상행동" ? getAbnormalDisplayData(data) : null;

  const level =
    title === "이상행동" ? abnormalData.level : data?.level || "idle";

  const state =
    title === "이상행동" ? abnormalData.state : data?.state || "대기";

  const score =
    title === "이상행동"
      ? safeNumber(abnormalData.risk_score)
      : safeNumber(data?.risk_score);

  const features = data?.features || {};

  return (
    <section className={`integrated-module-card ${level}`}>
      <div className="integrated-module-head">
        <div>
          <p>{title}</p>
          <h3>{state}</h3>
        </div>
        <StatusBadge level={level} />
      </div>

      <div className="integrated-module-score">
        <strong>{Math.round(score)}</strong>
        <span>위험 점수</span>
      </div>

      <p className="integrated-module-reason">
        {title === "이상행동"
          ? abnormalData.reason || "CSV 업로드 후 결과가 표시됩니다."
          : data?.reason || "CSV 업로드 후 결과가 표시됩니다."}
      </p>

      {title === "낙상 감지" && (
        <div className="integrated-mini-feature-grid">
          <div>
            <span>낙상 확률</span>
            <strong>{formatPercent(features.fall_prob)}</strong>
          </div>

          <div>
            <span>최대 속도</span>
            <strong>{formatSpeed(features.speed_max)}</strong>
          </div>

          <div>
            <span>높이 변화</span>
            <strong>{formatCm(features.height_drop)}</strong>
          </div>

          <div>
            <span>이후 이동</span>
            <strong>{formatCm(features.movement_after)}</strong>
          </div>
        </div>
      )}

      {title === "이상행동" && (
        <div className="integrated-mini-feature-grid">
          <div>
            <span>스트레스 점수</span>
            <strong>
              {abnormalData.stress_score !== null &&
              abnormalData.stress_score !== undefined
                ? `${Math.round(abnormalData.stress_score)}점`
                : "-"}
            </strong>
          </div>

          <div>
            <span>심박</span>
            <strong>
              {abnormalData.heart_rate !== null &&
              abnormalData.heart_rate !== undefined
                ? `${abnormalData.heart_rate} bpm`
                : "-"}
            </strong>
          </div>

          <div>
            <span>호흡</span>
            <strong>
              {abnormalData.respiratory_rate !== null &&
              abnormalData.respiratory_rate !== undefined
                ? `${abnormalData.respiratory_rate} rpm`
                : "-"}
            </strong>
          </div>

          <div>
            <span>체온</span>
            <strong>
              {abnormalData.temperature !== null &&
              abnormalData.temperature !== undefined
                ? `${abnormalData.temperature} ℃`
                : "-"}
            </strong>
          </div>
        </div>
      )}

      {title === "호흡" && (
        <div className="integrated-mini-feature-grid vital-simple">
          <div>
            <span>호흡 판단</span>
            <strong>{getVitalDisplay(data).label}</strong>
          </div>

          <div>
            <span>상태 라벨</span>
            <strong>{data?.condition || data?.state || "-"}</strong>
          </div>
        </div>
      )}
    </section>
  );
}

function VitalStatusCard({ data }) {
  const display = getVitalDisplay(data);

  return (
    <section className={`integrated-vital-status-card ${display.level}`}>
      <div className="integrated-vital-status-head">
        <div>
          <p className="integrated-kicker dark">호흡 위험도</p>
          <h2>호흡 상태</h2>
        </div>
        <StatusBadge level={display.level} text={display.badgeText} />
      </div>

      <div className="integrated-vital-status-body">
        <span>현재 판단</span>
        <strong>{display.label}</strong>
        <p>{display.detail}</p>
      </div>
    </section>
  );
}

function FallTimelineChart({ history, status }) {
  const ordered = [...history].reverse();

  const values = ordered.map((item) => ({
    second: safeNumber(item?.second),
    score: safeNumber(item?.fall?.risk_score),
    level: item?.fall?.level || "idle",
  }));

  if (values.length < 2) {
    return (
      <section className="integrated-chart-card compact">
        <div className="integrated-chart-title-row compact">
          <div>
            <h3>프레임별 낙상 판정</h3>
            <p>낙상 위험도 흐름</p>
          </div>
          <StatusBadge level="idle" text="대기" />
        </div>
        <div className="integrated-chart-empty compact">
          그래프 데이터 대기 중
        </div>
      </section>
    );
  }

  const width = 620;
  const height = 210;
  const padding = 32;

  const lastSecond = values[values.length - 1].second;

  const maxSecond = Math.max(
    safeNumber(status?.current_seconds),
    lastSecond,
    Math.max(...values.map((item) => item.second)),
    1,
  );

  const xBySecond = (second) => {
    return padding + (second / Math.max(maxSecond, 1)) * (width - padding * 2);
  };

  const yByScore = (score) => {
    return height - padding - (score / 100) * (height - padding * 2);
  };

  const points = values
    .map((item) => `${xBySecond(item.second)},${yByScore(item.score)}`)
    .join(" ");

  const dangerValues = values.filter((item) => item.score >= 70);
  const dangerStart = dangerValues.length ? dangerValues[0].second : null;
  const dangerEnd = dangerValues.length
    ? dangerValues[dangerValues.length - 1].second
    : null;

  const dangerY = yByScore(70);
  const maxScore = Math.max(...values.map((item) => item.score));

  return (
    <section className="integrated-chart-card compact">
      <div className="integrated-chart-title-row compact">
        <div>
          <h3>프레임별 낙상 판정</h3>
          <p>10프레임을 1초로 보고 위험 구간을 표시합니다.</p>
        </div>
        <StatusBadge
          level={maxScore >= 70 ? "danger" : "normal"}
          text={maxScore >= 70 ? "기준 초과" : "정상"}
        />
      </div>

      <div className="integrated-line-chart-wrap compact">
        <svg
          className="integrated-line-svg compact"
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
        >
          <line
            x1={padding}
            y1={height - padding}
            x2={width - padding}
            y2={height - padding}
            className="integrated-axis-line"
          />
          <line
            x1={padding}
            y1={padding}
            x2={padding}
            y2={height - padding}
            className="integrated-axis-line"
          />

          {dangerStart !== null && (
            <>
              <rect
                x={xBySecond(dangerStart)}
                y={padding}
                width={Math.max(
                  8,
                  xBySecond(dangerEnd + 1) - xBySecond(dangerStart),
                )}
                height={height - padding * 2}
                className="integrated-fall-zone"
              />
              <line
                x1={xBySecond(dangerStart)}
                y1={padding}
                x2={xBySecond(dangerStart)}
                y2={height - padding}
                className="integrated-fall-zone-line"
              />
              <text
                x={Math.min(xBySecond(dangerStart) + 8, width - 130)}
                y={padding + 14}
                className="integrated-fall-zone-text"
              >
                낙상 위험
              </text>
            </>
          )}

          <line
            x1={padding}
            y1={dangerY}
            x2={width - padding}
            y2={dangerY}
            className="integrated-threshold-line"
          />

          <text
            x={padding + 8}
            y={dangerY - 8}
            className="integrated-threshold-text"
          >
            기준 70점
          </text>

          <polyline points={points} className="integrated-score-line compact" />

          {values.map((item, index) => (
            <circle
              key={`${item.second}-${index}`}
              cx={xBySecond(item.second)}
              cy={yByScore(item.score)}
              r="3.5"
              className={
                item.score >= 70
                  ? "integrated-score-dot danger"
                  : "integrated-score-dot"
              }
            />
          ))}

          <text x={padding} y={height - 7} className="integrated-axis-text">
            0s
          </text>
          <text
            x={width - padding - 42}
            y={height - 7}
            className="integrated-axis-text"
          >
            {Math.round(maxSecond)}s
          </text>
        </svg>
      </div>

      <div className="integrated-chart-bottom compact">
        <span>{Math.round(lastSecond)}s까지 분석</span>
        <strong>최고 {Math.round(maxScore)}점</strong>
      </div>
    </section>
  );
}

function BehaviorTimeline({ history, status }) {
  const segments = useMemo(
    () => buildBehaviorSegments(history, status),
    [history, status],
  );

  const latestItem = history?.[0];
  const currentLabel = getBehaviorLabel(latestItem);
  const currentLevel = getBehaviorLevel(latestItem);

  const readItems = useMemo(() => {
    return [...(history || [])].reverse();
  }, [history]);

  const segmentMaxSecond = segments.length
    ? Math.max(...segments.map((item) => item.end || 0))
    : 1;

  const totalSeconds = Math.max(
    getDisplayTotalSeconds(status),
    safeNumber(status?.total_seconds),
    safeNumber(status?.abnormal_total_seconds),
    segmentMaxSecond,
    safeNumber(status?.current_seconds),
    1,
  );

  if (!history || history.length === 0) {
    return (
      <section className="integrated-behavior-card compact">
        <div className="integrated-chart-title-row compact">
          <div>
            <h3>행동 상태 타임라인</h3>
            <p>CSV Estimation 또는 eldercare 모델 결과를 표시합니다.</p>
          </div>
          <StatusBadge level="idle" text="대기" />
        </div>
        <div className="integrated-chart-empty compact">
          행동 데이터 대기 중
        </div>
      </section>
    );
  }

  return (
    <section className="integrated-behavior-card compact">
      <div className="integrated-chart-title-row compact">
        <div>
          <h3>행동 상태 타임라인</h3>
          <p>이상행동 상태를 시간 순서대로 표시합니다.</p>
        </div>
        <StatusBadge level={currentLevel} text={`현재 ${currentLabel}`} />
      </div>

      <div className="integrated-behavior-current-row">
        <strong>현재 이상행동 라벨</strong>
        <span
          className={`integrated-behavior-current ${getBehaviorClass(
            currentLabel,
          )}`}
        >
          {currentLabel}
        </span>
      </div>

      <div className="integrated-behavior-timeline compact">
        <div className="integrated-behavior-track compact">
          {segments.map((segment, index) => {
            const left = clamp(
              (segment.start / Math.max(totalSeconds, 1)) * 100,
              0,
              100,
            );

            const duration = Math.max(0.5, segment.end - segment.start);
            const rawWidth = (duration / Math.max(totalSeconds, 1)) * 100;
            const width = Math.max(1, Math.min(rawWidth, 100 - left));

            return (
              <div
                key={`${segment.label}-${segment.start}-${index}`}
                className={`integrated-behavior-segment ${segment.level} ${
                  segment.labelClass
                } ${segment.guardianAlert ? "guardian" : ""}`}
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                }}
                title={`${segment.start}s ~ ${segment.end}s / ${segment.label} / ${segment.riskScore}점`}
              >
                <span>{segment.label}</span>
                {segment.guardianAlert && <b>알림</b>}
              </div>
            );
          })}
        </div>

        <div className="integrated-behavior-axis">
          <span>0s</span>
          <span>{Math.round(totalSeconds / 2)}s</span>
          <span>{Math.round(totalSeconds)}s</span>
        </div>
      </div>

      <div className="integrated-behavior-legend compact">
        <span>
          <i className="etc" />
          기타
        </span>
        <span>
          <i className="sleep" />
          수면
        </span>
        <span>
          <i className="meal" />
          식사
        </span>
        <span>
          <i className="out" />
          외출
        </span>
        <span>
          <i className="warning" />
          주의
        </span>
        <span>
          <i className="danger" />
          위험
        </span>
      </div>
    </section>
  );
}

function FallReasonPanel({ bestFall, finalSummary }) {
  const fall = bestFall?.fall;
  const features = fall?.features || {};

  const risk = finalSummary?.fall_detected
    ? safeNumber(fall?.risk_score, finalSummary?.risk_score || 0)
    : safeNumber(fall?.risk_score);

  const heightDrop = safeNumber(features.height_drop);
  const speedMax = safeNumber(features.speed_max);
  const movementAfter = safeNumber(features.movement_after);

  const caseText = finalSummary?.fall_detected
    ? finalSummary?.fall_action || fall?.fall_action || "낙상 발생"
    : "낙상 없음";

  const summaryText = finalSummary?.fall_detected
    ? `${caseText} 가능성이 가장 높습니다. 속도 변화와 높이 변화, 이후 움직임을 함께 고려했습니다.`
    : "현재까지 낙상 기준을 넘는 위험 구간은 없습니다.";

  return (
    <section className="integrated-analysis-card">
      <div className="integrated-analysis-head">
        <div>
          <h2>낙상 행동 패턴 및 원인 분석</h2>
          <p>
            낙상 위험 구간, 속도, 높이 변화, 이후 움직임을 기반으로 원인을
            추정합니다.
          </p>
        </div>
        <StatusBadge
          level={finalSummary?.fall_detected ? "danger" : "normal"}
          text={finalSummary?.fall_detected ? caseText : "정상"}
        />
      </div>

      <div className="integrated-analysis-summary">
        <strong>{summaryText}</strong>
        <p>
          {finalSummary?.fall_cause ||
            fall?.fall_cause ||
            "CSV 재생이 진행되면 낙상 원인 분석이 표시됩니다."}
        </p>
      </div>

      <div className="integrated-analysis-grid">
        <div>
          <span>행동 케이스</span>
          <strong>{caseText}</strong>
        </div>
        <div>
          <span>추정 방향</span>
          <strong>
            {finalSummary?.fall_direction || fall?.fall_direction || "-"}
          </strong>
        </div>
        <div>
          <span>추정 원인</span>
          <strong>
            {finalSummary?.cause_guess || fall?.cause_guess || "-"}
          </strong>
        </div>
      </div>

      <div className="integrated-reason-split">
        <div className="integrated-reason-box">
          <h3>판단 근거</h3>
          <ul>
            <li>
              낙상 위험도 {Math.round(risk)}점으로 기준 70점과 비교합니다.
            </li>
            <li>높이 변화는 {formatCm(heightDrop)}입니다.</li>
            <li>최대 속도는 {formatSpeed(speedMax)}입니다.</li>
            <li>
              이후 이동거리는 {formatCm(movementAfter)}로 후속 움직임 감소
              여부를 확인합니다.
            </li>
          </ul>
        </div>

        <div className="integrated-reason-box">
          <h3>대응 권장</h3>
          <ul>
            <li>보호자 또는 관리자 확인이 필요합니다.</li>
            <li>낙상 구간 전후의 센서 로그를 확인하세요.</li>
            <li>
              후속 움직임이 거의 없다면 즉시 연락 또는 방문 확인이 필요합니다.
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}

function FallEvidenceCard({ bestFall, finalSummary }) {
  const fall = bestFall?.fall;
  const features = fall?.features || {};

  const risk = finalSummary?.fall_detected
    ? safeNumber(fall?.risk_score, finalSummary?.risk_score || 0)
    : safeNumber(fall?.risk_score);

  const heightDrop = safeNumber(features.height_drop);
  const speedMax = safeNumber(features.speed_max);
  const movementAfter = safeNumber(features.movement_after);

  const heightLevel =
    heightDrop >= 0.7 ? "danger" : heightDrop >= 0.4 ? "warning" : "normal";

  const speedLevel =
    speedMax >= 1.2 ? "danger" : speedMax >= 0.8 ? "warning" : "normal";

  const movementLevel =
    finalSummary?.fall_detected && movementAfter <= 0.2
      ? "danger"
      : movementAfter <= 0.5
        ? "warning"
        : "normal";

  return (
    <section className="integrated-side-card">
      <div className="integrated-panel-head">
        <div>
          <p className="integrated-kicker dark">낙상 판단 근거</p>
          <h2>근거 그래프</h2>
        </div>
      </div>

      <div className="integrated-evidence-list">
        <RiskBar
          label="낙상 위험도"
          value={risk}
          valueText={`${Math.round(risk)}점`}
          percent={risk}
          level={risk >= 70 ? "danger" : risk >= 40 ? "warning" : "normal"}
        />

        <RiskBar
          label="높이 변화"
          value={heightDrop}
          valueText={formatCm(heightDrop)}
          percent={Math.min(100, heightDrop * 120)}
          level={heightLevel}
        />

        <RiskBar
          label="최대 속도"
          value={speedMax}
          valueText={formatSpeed(speedMax)}
          percent={Math.min(100, speedMax * 80)}
          level={speedLevel}
        />

        <RiskBar
          label="이후 이동거리"
          value={movementAfter}
          valueText={formatCm(movementAfter)}
          percent={Math.min(100, movementAfter * 100)}
          level={movementLevel}
        />
      </div>

      <div className="integrated-why-box">
        <strong>낙상으로 판단한 이유</strong>
        <p>
          {finalSummary?.fall_detected
            ? `최종 판단: ${finalSummary.label}. ${finalSummary.fall_cause}`
            : "아직 낙상으로 판단된 위험 구간이 없습니다."}
        </p>
      </div>
    </section>
  );
}

function RealtimeLogCard({ history }) {
  const [page, setPage] = useState(1);
  const totalPage = Math.max(1, Math.ceil(history.length / PAGE_SIZE));

  useEffect(() => {
    if (page > totalPage) setPage(totalPage);
  }, [history.length, page, totalPage]);

  const visible = history.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <section className="integrated-side-card">
      <div className="integrated-panel-head">
        <div>
          <p className="integrated-kicker dark">실시간 통합 로그</p>
          <h2>최근 판정</h2>
        </div>
        <span className="integrated-count-chip">5개씩</span>
      </div>

      <div className="integrated-compact-list">
        {visible.length === 0 ? (
          <div className="integrated-empty small">실행 로그가 없습니다.</div>
        ) : (
          visible.map((item, index) => {
            const abnormalLabel = getBehaviorLabel(item);

            return (
              <article
                className={`integrated-compact-item ${item.overall.level}`}
                key={`${item.step}-${index}`}
              >
                <div className="integrated-compact-top">
                  <strong>{item.second}s</strong>
                  <StatusBadge
                    level={item.overall.level}
                    text={item.overall.label}
                  />
                </div>
                <p>
                  낙상 {item.fall.state} · 이상행동 {abnormalLabel}
                </p>
                <span>{formatTime(item.time)}</span>
              </article>
            );
          })
        )}
      </div>

      <div className="integrated-pagination">
        <button
          disabled={page <= 1}
          onClick={() => setPage((prev) => prev - 1)}
        >
          &lt;
        </button>
        <strong>
          {page} / {totalPage}
        </strong>
        <button
          disabled={page >= totalPage}
          onClick={() => setPage((prev) => prev + 1)}
        >
          &gt;
        </button>
      </div>
    </section>
  );
}

function SavedLogCard({ savedLogs, dbInfo, onClear }) {
  const [page, setPage] = useState(1);
  const totalPage = Math.max(1, Math.ceil(savedLogs.length / PAGE_SIZE));

  useEffect(() => {
    if (page > totalPage) setPage(totalPage);
  }, [savedLogs.length, page, totalPage]);

  const visible = savedLogs.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <section className="integrated-side-card">
      <div className="integrated-panel-head">
        <div>
          <p className="integrated-kicker dark">DB 저장 로그</p>
          <h2>저장 이벤트</h2>
        </div>
        <StatusBadge
          level={dbInfo?.db_connected ? "normal" : "warning"}
          text={dbInfo?.db_connected ? "MongoDB" : "Memory"}
        />
      </div>

      <button className="integrated-delete-btn" onClick={onClear}>
        저장 로그 전체 삭제
      </button>

      <div className="integrated-compact-list">
        {visible.length === 0 ? (
          <div className="integrated-empty small">
            저장된 이벤트가 없습니다.
          </div>
        ) : (
          visible.map((item, index) => {
            const summary = item.final_summary;
            const result = item.result;

            const label =
              summary?.label ||
              result?.overall?.label ||
              item.event_type ||
              "저장 이벤트";

            const second = summary?.fall_second ?? result?.second ?? "-";

            return (
              <article
                className="integrated-saved-item"
                key={`${item._id || index}`}
              >
                <div className="integrated-compact-top">
                  <strong>{label}</strong>
                  <span>{second !== "-" ? `${second}s` : "-"}</span>
                </div>
                <p>
                  {summary?.message ||
                    result?.fall?.fall_cause ||
                    result?.overall?.message ||
                    "저장된 통합 이벤트입니다."}
                </p>
                <span>{formatTime(item.saved_at)}</span>
              </article>
            );
          })
        )}
      </div>

      <div className="integrated-pagination">
        <button
          disabled={page <= 1}
          onClick={() => setPage((prev) => prev - 1)}
        >
          &lt;
        </button>
        <strong>
          {page} / {totalPage}
        </strong>
        <button
          disabled={page >= totalPage}
          onClick={() => setPage((prev) => prev + 1)}
        >
          &gt;
        </button>
      </div>
    </section>
  );
}

export default function IntegratedDashboard() {
  const timerRef = useRef(null);

  const [fps, setFps] = useState(10);
  const [fallWindowFrames, setFallWindowFrames] = useState(10);

  const [fallFile, setFallFile] = useState(null);
  const [abnormalFile, setAbnormalFile] = useState(null);
  const [abnormalCsvMeta, setAbnormalCsvMeta] = useState({
    rowCount: 0,
    estimationColumn: null,
    labels: [],
    error: null,
  });
  const [vitalFile, setVitalFile] = useState(null);

  const [sessionId, setSessionId] = useState("");
  const [status, setStatus] = useState(null);
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const [finalSummary, setFinalSummary] = useState(null);
  const [savedLogs, setSavedLogs] = useState([]);
  const [dbInfo, setDbInfo] = useState(null);

  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [uploadPanelOpen, setUploadPanelOpen] = useState(true);

  const displaySummary = finalSummary || {
    level: "idle",
    label: "대기",
    risk_score: 0,
    message: "CSV를 업로드하고 통합 재생을 시작하세요.",
    fall_detected: false,
    saved: false,
  };

  const displayStatus = useMemo(() => {
    const rowCount = safeNumber(abnormalCsvMeta.rowCount, 0);

    if (!status || rowCount <= 0) return status;

    return {
      ...status,
      total_steps: rowCount,
      total_seconds: rowCount,
      abnormal_row_count: rowCount,
      abnormal_total_seconds: rowCount,
      profile: {
        ...(status.profile || {}),
        abnormal_rows: rowCount,
        abnormal_total_seconds: rowCount,
      },
    };
  }, [status, abnormalCsvMeta.rowCount]);

  const displayTotalSeconds = useMemo(() => {
    return getDisplayTotalSeconds(displayStatus);
  }, [displayStatus]);

  const displayCurrentSeconds = useMemo(() => {
    return getDisplayCurrentSeconds(displayStatus);
  }, [displayStatus]);

  const progress = useMemo(() => {
    if (!displayTotalSeconds) return 0;
    return clamp((displayCurrentSeconds / displayTotalSeconds) * 100, 0, 100);
  }, [displayCurrentSeconds, displayTotalSeconds]);

  const bestFall = useMemo(() => findBestFall(history), [history]);
  const bestVital = useMemo(() => findBestModule(history, "vital"), [history]);

  const finalFallForCard = useMemo(() => {
    if (displaySummary?.fall_detected && bestFall?.fall) {
      return {
        ...bestFall.fall,
        state: "Fall Alert",
        level: "danger",
      };
    }

    return current?.fall;
  }, [displaySummary, bestFall, current]);

  const finalAbnormalForCard = useMemo(() => {
    return getAbnormalDisplayData(current?.abnormal);
  }, [current]);

  const finalVitalForCard = useMemo(() => {
    if (
      bestVital?.vital?.level === "danger" ||
      bestVital?.vital?.level === "warning"
    ) {
      return bestVital.vital;
    }

    return current?.vital;
  }, [bestVital, current]);

  const abnormalRiskValue =
    finalAbnormalForCard?.state === "대기"
      ? 0
      : abnormalScoreFromLabel(finalAbnormalForCard?.state);

  function getApiErrorMessage(data, fallback = "API 요청 실패") {
    if (!data) return fallback;

    if (typeof data.detail === "string") return data.detail;

    if (data.detail && typeof data.detail === "object") {
      if (data.detail.message) return data.detail.message;
      return JSON.stringify(data.detail);
    }

    if (typeof data.message === "string") return data.message;

    return fallback;
  }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, options);

    let data = null;

    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }

    if (!response.ok) {
      throw new Error(getApiErrorMessage(data));
    }

    return data;
  }

  async function loadSavedLogs() {
    try {
      const data = await request("/integrated/saved-events?limit=50");
      setSavedLogs(data.items || []);
      setDbInfo(data);
    } catch (error) {
      setDbInfo({
        db_connected: false,
        db_error: error.message,
      });
    }
  }

  async function clearSavedLogs() {
    const ok = window.confirm("DB 저장 로그를 전체 삭제할까요?");

    if (!ok) return;

    try {
      await request("/integrated/saved-events", {
        method: "DELETE",
      });

      setMessage("DB 저장 로그가 삭제되었습니다.");
      await loadSavedLogs();
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function handleAbnormalFileChange(file) {
    setAbnormalFile(file);

    if (!file) {
      setAbnormalCsvMeta({
        rowCount: 0,
        estimationColumn: null,
        labels: [],
        error: null,
      });
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setAbnormalCsvMeta({
        rowCount: 0,
        estimationColumn: null,
        labels: [],
        error: "브라우저 직접 Estimation 파싱은 CSV에서만 지원합니다.",
      });
      return;
    }

    try {
      const text = await file.text();
      const meta = extractEstimationLabelsFromCsvText(text);
      setAbnormalCsvMeta(meta);
    } catch (error) {
      setAbnormalCsvMeta({
        rowCount: 0,
        estimationColumn: null,
        labels: [],
        error: error.message,
      });
    }
  }

  async function uploadAll() {
    if (!fallFile && !abnormalFile && !vitalFile) {
      setMessage("최소 1개 이상의 CSV 파일을 선택해야 합니다.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const formData = new FormData();

      if (fallFile) formData.append("fall_csv", fallFile);
      if (abnormalFile) formData.append("abnormal_csv", abnormalFile);
      if (vitalFile) formData.append("vital_csv", vitalFile);

      formData.append("fps", String(fps));
      formData.append("fall_window_frames", String(fallWindowFrames));

      const data = await request("/integrated/simulation/upload", {
        method: "POST",
        body: formData,
      });

      setSessionId(data.status.session_id);
      setStatus(data.status);
      setCurrent(null);
      setHistory([]);
      setFinalSummary(data.final_summary);
      setRunning(false);
      setMessage("통합 CSV 업로드가 완료되었습니다.");
      await loadSavedLogs();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  async function nextStep() {
    if (!sessionId) return;

    try {
      const data = await request(`/integrated/simulation/${sessionId}/next`);

      const patchedResult = patchResultWithClientEstimation(
        data.result,
        abnormalCsvMeta.labels,
      );
      const patchedHistory = (data.history || []).map((item) =>
        patchResultWithClientEstimation(item, abnormalCsvMeta.labels),
      );

      if (patchedResult) {
        setCurrent(patchedResult);
      }

      setStatus(data.status);
      setHistory(patchedHistory);
      setFinalSummary(data.final_summary || null);

      if (data.done) {
        setRunning(false);
        await loadSavedLogs();
      }

      if (data.result?.db_saved) {
        await loadSavedLogs();
      }
    } catch (error) {
      setRunning(false);
      setMessage(error.message);
    }
  }

  async function resetSimulation() {
    if (!sessionId) return;

    setLoading(true);

    try {
      const data = await request(`/integrated/simulation/${sessionId}/reset`, {
        method: "POST",
      });

      setStatus(data.status);
      setCurrent(null);
      setHistory([]);
      setFinalSummary(data.final_summary);
      setRunning(false);
      setMessage("통합 시뮬레이션이 초기화되었습니다.");
      await loadSavedLogs();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleRun() {
    if (!sessionId) {
      setMessage("먼저 CSV를 업로드해야 합니다.");
      return;
    }

    setRunning((prev) => !prev);
  }

  useEffect(() => {
    loadSavedLogs();
  }, []);

  useEffect(() => {
    if (!running || !sessionId) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      return;
    }

    nextStep();

    timerRef.current = setInterval(() => {
      nextStep();
    }, PLAY_INTERVAL_MS);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [running, sessionId]);

  return (
    <main className="integrated-page">
      <section className={`integrated-hero ${displaySummary.level}`}>
        <div>
          <p className="integrated-kicker">Smart Care AI 통합 관제</p>
          <h1>낙상 · 이상행동 통합 관제</h1>
          <p>
            낙상 RF 모델, 이상행동 eldercare 모델, 호흡 모델을 각각 실행해 통합
            표시합니다.
          </p>
        </div>

        <div className="integrated-hero-result">
          <span>최종 종합 상태</span>
          <strong>{displaySummary.label}</strong>
          <p>{displaySummary.message}</p>
        </div>
      </section>

      <section
        className={`integrated-upload-panel ${
          uploadPanelOpen ? "open" : "collapsed"
        }`}
      >
        <div className="integrated-upload-header">
          <button
            type="button"
            className="integrated-upload-title-button"
            onClick={() => setUploadPanelOpen((prev) => !prev)}
          >
            <p className="integrated-kicker dark">CSV 통합 입력</p>
            <h2>파일 한 번에 업로드</h2>

            {!uploadPanelOpen && (
              <span className="integrated-upload-closed-summary">
                낙상 {fallFile ? "선택됨" : "대기"} · 이상행동{" "}
                {abnormalFile ? "선택됨" : "대기"} · 호흡{" "}
                {vitalFile ? "선택됨" : "대기"}
              </span>
            )}
          </button>

          <div className="integrated-upload-header-actions">
            <button
              type="button"
              className="integrated-collapse-btn"
              onClick={() => setUploadPanelOpen((prev) => !prev)}
              aria-label={
                uploadPanelOpen ? "업로드 패널 닫기" : "업로드 패널 열기"
              }
              aria-expanded={uploadPanelOpen}
            >
              <span
                className={`integrated-chevron-icon ${
                  uploadPanelOpen ? "open" : ""
                }`}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M6 9L12 15L18 9"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
            </button>
          </div>
        </div>

        {uploadPanelOpen && (
          <div className="integrated-upload-body">
            <div className="integrated-upload-grid main-only">
              <UploadBox
                title="낙상 CSV"
                desc="frame, x, y, z, v / 낙상은 fps 기준"
                file={fallFile}
                onChange={setFallFile}
              />

              <UploadBox
                title="이상행동 CSV"
                desc={
                  abnormalCsvMeta.rowCount > 0
                    ? `Estimation 직접 읽기 · ${abnormalCsvMeta.rowCount}행 = ${abnormalCsvMeta.rowCount}초`
                    : "Estimation 컬럼 직접 읽기 / 1행 = 1초"
                }
                file={abnormalFile}
                onChange={handleAbnormalFileChange}
                note={
                  abnormalCsvMeta.error
                    ? abnormalCsvMeta.error
                    : abnormalCsvMeta.estimationColumn
                      ? `읽은 컬럼: ${abnormalCsvMeta.estimationColumn}`
                      : null
                }
              />
            </div>

            <div className="integrated-vital-upload-mini-row">
              <UploadBox
                title="호흡 CSV"
                desc="feature CSV 또는 VitalSignal 원본 CSV"
                file={vitalFile}
                onChange={setVitalFile}
                compact
                note="통합 화면에는 이상 여부만 별도 표시"
              />
            </div>

            <div className="integrated-control-row">
              <button onClick={uploadAll} disabled={loading}>
                {loading ? "업로드 중..." : "통합 CSV 업로드"}
              </button>

              <button onClick={toggleRun} disabled={!sessionId}>
                {running ? "일시정지" : "전체 재생"}
              </button>

              <button onClick={nextStep} disabled={!sessionId || running}>
                1초 넘기기
              </button>

              <button onClick={resetSimulation} disabled={!sessionId}>
                초기화
              </button>
            </div>

            {displayStatus?.profile && (
              <p className="integrated-message">
                코드 {displayStatus.code_version || CODE_VERSION} · 이상행동{" "}
                {displayStatus.profile.abnormal_rows || 0}행 ={" "}
                {displayTotalSeconds}초 · 전체 {displayStatus.total_steps || 0}{" "}
                step / 화면 기준 {displayTotalSeconds}초
              </p>
            )}

            {message && <p className="integrated-message">{message}</p>}
          </div>
        )}
      </section>

      <section className="integrated-progress-panel">
        <div className="integrated-progress-top">
          <span>
            진행 시간 {displayCurrentSeconds}s / {displayTotalSeconds}s
          </span>
          <strong>{Math.round(progress)}%</strong>
        </div>
        <div className="integrated-progress-track">
          <div
            className={`integrated-progress-fill ${displaySummary.level}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      </section>

      <section className="integrated-main-grid">
        <div className="integrated-left">
          <div className="integrated-module-grid two-main">
            <ModuleCard title="낙상 감지" data={finalFallForCard} />
            <ModuleCard title="이상행동" data={finalAbnormalForCard} />
          </div>

          <div className="integrated-chart-pair-grid">
            <FallTimelineChart history={history} status={displayStatus} />
            <BehaviorTimeline history={history} status={displayStatus} />
          </div>

          <FallReasonPanel bestFall={bestFall} finalSummary={displaySummary} />
        </div>

        <aside className="integrated-right">
          <section className="integrated-side-card">
            <p className="integrated-kicker dark">종합 위험도</p>
            <h2>모듈별 점수</h2>

            <div className="integrated-risk-list">
              <RiskBar
                label="낙상 위험도"
                value={finalFallForCard?.risk_score || 0}
                valueText={`${Math.round(finalFallForCard?.risk_score || 0)}점`}
                percent={finalFallForCard?.risk_score || 0}
                level={finalFallForCard?.level || "idle"}
              />

              <RiskBar
                label="이상행동 위험도"
                value={abnormalRiskValue}
                valueText={`${Math.round(abnormalRiskValue)}점`}
                percent={abnormalRiskValue}
                level={finalAbnormalForCard?.level || "idle"}
              />
            </div>
          </section>

          <VitalStatusCard data={finalVitalForCard} />

          <FallEvidenceCard bestFall={bestFall} finalSummary={displaySummary} />

          <RealtimeLogCard history={history} />

          <SavedLogCard
            savedLogs={savedLogs}
            dbInfo={dbInfo}
            onClear={clearSavedLogs}
          />
        </aside>
      </section>
    </main>
  );
}
