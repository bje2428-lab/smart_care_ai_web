import { useEffect, useMemo, useRef, useState } from "react";
import {
  getVitalHealth,
  getVitalStats,
  getVitalHistory,
  uploadVitalSimulationCsv,
  getVitalSimulationStatus,
  resetVitalSimulation,
  clearVitalSimulation,
  nextVitalSimulation,
  VITAL_API_URL,
} from "../api/vitalApi";

import "./VitalDashboard.css";

function getStatusClass(status) {
  if (status === "Danger") return "danger";
  if (status === "Warning") return "warning";
  if (status === "Normal") return "normal";
  return "unknown";
}

function statusText(status, state) {
  if (state) return state;
  if (status === "Danger") return "위험";
  if (status === "Warning") return "주의";
  if (status === "Normal") return "정상";
  return "대기";
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";

  const number = Number(value);

  if (Number.isNaN(number)) return value;

  return number.toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined) return "-";

  const number = Number(value);

  if (Number.isNaN(number)) return "-";

  return `${Math.round(number * 100)}%`;
}

function inputModeText(mode, description) {
  if (description) return description;
  if (mode === "feature_csv") return "전처리 feature CSV";
  if (mode === "raw_signal_csv") return "원본 VitalSignal CSV";
  if (mode === "numeric_feature_fallback") return "숫자 컬럼 자동 매핑";
  return "자동 감지 대기";
}

function VitalDashboard() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);

  const [simulation, setSimulation] = useState(null);
  const [result, setResult] = useState(null);
  const [segments, setSegments] = useState([]);

  const [selectedFile, setSelectedFile] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [segmentPage, setSegmentPage] = useState(1);

  const timerRef = useRef(null);

  const segmentPageSize = 5;
  const segmentTotalPages = Math.max(
    1,
    Math.ceil(segments.length / segmentPageSize),
  );

  const currentSegments = useMemo(() => {
    const start = (segmentPage - 1) * segmentPageSize;
    return segments.slice(start, start + segmentPageSize);
  }, [segments, segmentPage]);

  const recentChartItems = useMemo(() => {
    return history.slice(0, 5).map((item) => ({
      label: `${item.risk_score ?? 0}%`,
      value: Math.max(0, Math.min(Number(item.risk_score || 0), 100)),
      status: item.status,
    }));
  }, [history]);

  const latestSegment = segments[segments.length - 1];
  const displayStatus = latestSegment?.status || result?.status;
  const displayState = latestSegment?.state || result?.state;
  const displayClass = getStatusClass(displayStatus);

  const loadDashboardData = async () => {
    setHealthLoading(true);

    try {
      const [healthData, statsData, historyData] = await Promise.all([
        getVitalHealth(),
        getVitalStats(),
        getVitalHistory(20),
      ]);

      setHealth(healthData);
      setStats(statsData);
      setHistory(historyData?.history || []);

      try {
        const simData = await getVitalSimulationStatus();
        setSimulation(simData);

        if (simData?.results?.length > 0) {
          setSegments(simData.results);
        }
      } catch (error) {
        // 시뮬레이션 상태 확인 실패는 전체 화면 오류로 막지 않음
      }
    } catch (error) {
      setMessage(error.message || "생체신호 API 연결에 실패했습니다.");
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isPlaying) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = setInterval(async () => {
      try {
        const data = await nextVitalSimulation();

        if (data?.success === false) {
          setMessage(data.message || "다음 구간 분석에 실패했습니다.");
          setIsPlaying(false);
          return;
        }

        if (data?.segment) {
          setSegments(data.results || []);
          setResult(data.summary || null);
          setSimulation({
            loaded: true,
            file_name: data.file_name,
            input_mode: data.input_mode || data.summary?.input_mode,
            input_description:
              data.input_description || data.summary?.input_description,
            total_segments: data.total_segments,
            current_index: data.current_index,
            remaining_segments: data.remaining_segments,
            finished: data.finished,
          });

          const nextPage = Math.max(
            1,
            Math.ceil((data.results || []).length / segmentPageSize),
          );
          setSegmentPage(nextPage);
        }

        if (data?.finished) {
          setIsPlaying(false);
          setMessage("생체신호 실시간 분석이 완료되었습니다.");
          await loadDashboardData();
        }
      } catch (error) {
        setMessage(
          error.message || "실시간 생체신호 분석 중 오류가 발생했습니다.",
        );
        setIsPlaying(false);
      }
    }, 1000);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPlaying]);

  const handleUploadSimulation = async () => {
    if (!selectedFile) {
      setMessage("CSV 파일을 먼저 선택해주세요.");
      return;
    }

    setLoading(true);
    setMessage("");
    setIsPlaying(false);

    try {
      const data = await uploadVitalSimulationCsv(selectedFile);

      if (data?.success === false) {
        setMessage(data.message || "CSV 업로드에 실패했습니다.");
        return;
      }

      setSegments([]);
      setResult(null);
      setSegmentPage(1);
      setSimulation({
        loaded: true,
        file_name: data.file_name,
        input_mode: data.input_mode,
        input_description: data.input_description,
        detected_columns: data.detected_columns,
        total_samples: data.total_samples,
        raw_total_rows: data.raw_total_rows,
        total_segments: data.total_segments,
        current_index: data.current_index,
        remaining_segments: data.remaining_segments,
        finished: false,
      });

      setMessage(
        "CSV 업로드 완료. 실시간 시작 버튼을 누르면 1초마다 분석됩니다.",
      );
      await loadDashboardData();
    } catch (error) {
      setMessage(error.message || "CSV 업로드에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleStart = () => {
    if (!simulation?.loaded) {
      setMessage("먼저 CSV 파일을 업로드해주세요.");
      return;
    }

    if (simulation?.finished) {
      setMessage(
        "이미 분석이 완료되었습니다. 다시 실행하려면 처음부터 버튼을 눌러주세요.",
      );
      return;
    }

    setMessage("생체신호 실시간 분석을 시작했습니다.");
    setIsPlaying(true);
  };

  const handleStop = () => {
    setIsPlaying(false);
    setMessage("생체신호 실시간 분석을 일시정지했습니다.");
  };

  const handleNextOnce = async () => {
    if (!simulation?.loaded) {
      setMessage("먼저 CSV 파일을 업로드해주세요.");
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const data = await nextVitalSimulation();

      if (data?.success === false) {
        setMessage(data.message || "다음 구간 분석에 실패했습니다.");
        return;
      }

      if (data?.segment) {
        setSegments(data.results || []);
        setResult(data.summary || null);
        setSimulation({
          loaded: true,
          file_name: data.file_name,
          total_segments: data.total_segments,
          current_index: data.current_index,
          remaining_segments: data.remaining_segments,
          finished: data.finished,
        });

        const nextPage = Math.max(
          1,
          Math.ceil((data.results || []).length / segmentPageSize),
        );
        setSegmentPage(nextPage);
      }

      if (data?.finished) {
        setIsPlaying(false);
        setMessage("생체신호 실시간 분석이 완료되었습니다.");
        await loadDashboardData();
      }
    } catch (error) {
      setMessage(error.message || "다음 구간 분석 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleResetSimulation = async () => {
    setLoading(true);
    setIsPlaying(false);
    setMessage("");

    try {
      await resetVitalSimulation();
      setSegments([]);
      setResult(null);
      setSegmentPage(1);
      setMessage("생체신호 실시간 분석을 처음부터 다시 시작할 수 있습니다.");
      await loadDashboardData();
    } catch (error) {
      setMessage(error.message || "시뮬레이션 초기화에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const handleClearSimulation = async () => {
    setLoading(true);
    setIsPlaying(false);
    setMessage("");

    try {
      await clearVitalSimulation();
      setSegments([]);
      setResult(null);
      setSimulation(null);
      setSelectedFile(null);
      setSegmentPage(1);
      setMessage("생체신호 실시간 분석 데이터를 삭제했습니다.");
      await loadDashboardData();
    } catch (error) {
      setMessage(error.message || "시뮬레이션 삭제에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page vital-dashboard-page">
      <header className="dashboard-header">
        <div>
          <h1>독거노인 생체신호 실시간 감지 대시보드</h1>
          <p>
            원본 VitalSignal CSV 또는 전처리 feature CSV를 업로드하면 1초 단위로
            자동 전처리·분석해 실시간 관제처럼 표시합니다.
          </p>
        </div>

        <button className="refresh-btn" onClick={loadDashboardData}>
          {healthLoading ? "확인 중..." : "연결 다시 확인"}
        </button>
      </header>

      <section className="fall-summary-grid">
        <article className="fall-summary-card">
          <p>API 상태</p>
          <strong>{health?.status === "ok" ? "정상 연결" : "대기"}</strong>
          <span>{VITAL_API_URL || "VITE_API_URL 미설정"}</span>
        </article>

        <article className="fall-summary-card">
          <p>생체신호 모델</p>
          <strong>{health?.model_loaded ? "로드 완료" : "로드 필요"}</strong>
          <span>
            {health?.model_file_exists
              ? "models 폴더에서 모델 파일을 찾았습니다."
              : "모델 파일을 확인해주세요."}
          </span>
        </article>

        <article className="fall-summary-card">
          <p>CSV 입력 방식</p>
          <strong>
            {inputModeText(
              simulation?.input_mode,
              simulation?.input_description,
            )}
          </strong>
          <span>
            원본 VitalSignal CSV와 전처리 feature CSV를 모두 지원합니다.
          </span>
        </article>

        <article className="fall-summary-card">
          <p>진행 구간</p>
          <strong>
            {simulation?.current_index ?? 0} / {simulation?.total_segments ?? 0}
          </strong>
          <span>1초 구간 기준 진행률</span>
        </article>

        <article className="fall-summary-card danger-soft">
          <p>보호자 알림</p>
          <strong>{stats?.alert_count ?? 0}</strong>
          <span>위험 상태로 판단된 누적 건수</span>
        </article>
      </section>

      {message && <p className="message">{message}</p>}

      <main className="main-grid">
        <section className="left-panel">
          <article className="card upload-card">
            <h2>CSV 생체신호 실시간 테스트</h2>
            <p className="desc">
              원본 CSV는 1초 단위 feature로 자동 전처리하고, 전처리 완료 CSV는
              그대로 1초 구간으로 묶어 실시간처럼 분석합니다.
            </p>

            <div className="upload-box vital-realtime-upload-box">
              <input
                type="file"
                accept=".csv"
                onChange={(event) =>
                  setSelectedFile(event.target.files?.[0] || null)
                }
              />

              <button
                type="button"
                onClick={handleUploadSimulation}
                disabled={loading}
              >
                CSV 업로드
              </button>

              <button
                type="button"
                onClick={isPlaying ? handleStop : handleStart}
                disabled={loading || !simulation?.loaded}
              >
                {isPlaying ? "일시정지" : "실시간 시작"}
              </button>

              <button
                type="button"
                onClick={handleNextOnce}
                disabled={loading || isPlaying || !simulation?.loaded}
              >
                1초 분석
              </button>

              <button
                type="button"
                onClick={handleResetSimulation}
                disabled={loading || !simulation?.loaded}
              >
                처음부터
              </button>

              <button
                type="button"
                onClick={handleClearSimulation}
                disabled={loading}
              >
                삭제
              </button>
            </div>

            <p className="selected-file">
              선택 파일: {selectedFile?.name || simulation?.file_name || "-"}
              {simulation?.loaded && (
                <>
                  {" "}
                  · 입력 방식:{" "}
                  {inputModeText(
                    simulation?.input_mode,
                    simulation?.input_description,
                  )}
                </>
              )}
            </p>

            <div className="realtime-panel">
              <div className="realtime-title-row">
                <div>
                  <h3>1초 단위 실시간 생체신호 분석</h3>
                  <p>
                    원본 신호는 먼저 1초 단위 feature로 변환한 뒤, 화면에는
                    5개씩 페이지로 나누어 표시합니다.
                  </p>
                </div>

                <span className={`live-chip ${isPlaying ? "on" : ""}`}>
                  {isPlaying
                    ? "실시간 분석 중"
                    : simulation?.finished
                      ? "완료"
                      : "대기 중"}
                </span>
              </div>

              <div className="realtime-grid">
                <div>
                  <span>분석 파일</span>
                  <strong>{simulation?.file_name || "-"}</strong>
                </div>

                <div>
                  <span>전체 구간</span>
                  <strong>{simulation?.total_segments ?? 0}</strong>
                </div>

                <div>
                  <span>현재 구간</span>
                  <strong>{simulation?.current_index ?? 0}</strong>
                </div>

                <div>
                  <span>현재 상태</span>
                  <strong>{statusText(displayStatus, displayState)}</strong>
                </div>
              </div>

              <div className="compact-live-summary">
                <span>
                  정상 구간: <strong>{result?.normal_count ?? 0}</strong>
                </span>
                <span>
                  주의 구간: <strong>{result?.warning_count ?? 0}</strong>
                </span>
                <span>
                  위험 구간: <strong>{result?.danger_count ?? 0}</strong>
                </span>
                <span>
                  최근 오차:{" "}
                  <strong>{formatNumber(latestSegment?.error_max, 6)}</strong>
                </span>
              </div>

              <div className="table-wrap realtime-table-wrap">
                <table className="history-log-table">
                  <thead>
                    <tr>
                      <th>구간</th>
                      <th>시간</th>
                      <th>상태</th>
                      <th>위험도</th>
                      <th>최대 오차</th>
                      <th>이상 비율</th>
                      <th>입력</th>
                    </tr>
                  </thead>

                  <tbody>
                    {currentSegments.length === 0 ? (
                      <tr>
                        <td colSpan="7" className="empty">
                          아직 실시간 분석 결과가 없습니다.
                        </td>
                      </tr>
                    ) : (
                      currentSegments.map((segment) => (
                        <tr key={segment.segment_index}>
                          <td>{segment.segment_index}</td>
                          <td>
                            {segment.start_second}s ~ {segment.end_second}s
                          </td>
                          <td>
                            <span
                              className={`badge ${getStatusClass(segment.status)}`}
                            >
                              {statusText(segment.status, segment.state)}
                            </span>
                          </td>
                          <td>{segment.risk_score}점</td>
                          <td>{formatNumber(segment.error_max, 6)}</td>
                          <td>{formatPercent(segment.anomaly_ratio)}</td>
                          <td>{inputModeText(segment.input_mode)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              <div className="live-pagination">
                <button
                  type="button"
                  disabled={segmentPage <= 1}
                  onClick={() =>
                    setSegmentPage((prev) => Math.max(1, prev - 1))
                  }
                >
                  &lt;
                </button>

                <span>
                  {segmentPage} / {segmentTotalPages}
                </span>

                <button
                  type="button"
                  disabled={segmentPage >= segmentTotalPages}
                  onClick={() =>
                    setSegmentPage((prev) =>
                      Math.min(segmentTotalPages, prev + 1),
                    )
                  }
                >
                  &gt;
                </button>
              </div>
            </div>

            {latestSegment && (
              <div className="result-box">
                <div className="result-header">
                  <div>
                    <h3>최근 1초 구간 판정</h3>
                    <p className="result-message">
                      {latestSegment.message || "최근 구간 분석 결과입니다."}
                    </p>
                  </div>

                  <span className={`badge ${displayClass}`}>
                    {statusText(displayStatus, displayState)}
                  </span>
                </div>

                <div className="result-grid">
                  <div>
                    <span>위험 점수</span>
                    <strong>{latestSegment.risk_score ?? 0}점</strong>
                  </div>

                  <div>
                    <span>최대 복원 오차</span>
                    <strong>{formatNumber(latestSegment.error_max, 6)}</strong>
                  </div>

                  <div>
                    <span>주의 기준</span>
                    <strong>
                      {formatNumber(
                        latestSegment.warning_threshold ||
                          latestSegment.threshold,
                        6,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>보호자 알림</span>
                    <strong
                      className={
                        latestSegment.alert ? "alert-true" : "alert-false"
                      }
                    >
                      {latestSegment.alert ? "알림 필요" : "미발송"}
                    </strong>
                  </div>
                </div>
              </div>
            )}
          </article>
        </section>

        <aside className="side-panel">
          <article className="chart-card history-chart-card">
            <div className="chart-title-row">
              <div>
                <h3>실시간 위험도 그래프</h3>
                <p>현재 실행 중인 1초 구간별 위험 점수입니다.</p>
              </div>
            </div>

            <div className="mini-column-chart">
              {segments.length === 0 ? (
                <p className="empty-chart-text">
                  표시할 위험도 데이터가 없습니다.
                </p>
              ) : (
                segments.slice(-5).map((item) => (
                  <div className="mini-column-item" key={item.segment_index}>
                    <div className="mini-column-wrap">
                      <div
                        className={`mini-column ${getStatusClass(item.status)}`}
                        style={{
                          height: `${Math.max(item.risk_score || 0, 5)}%`,
                        }}
                      />
                    </div>
                    <span>{item.risk_score ?? 0}%</span>
                  </div>
                ))
              )}
            </div>
          </article>

          <article className="card events-card">
            <div className="section-title-row">
              <div>
                <h2>최근 생체신호 로그</h2>
                <p className="desc">완료된 실시간 분석 이벤트가 표시됩니다.</p>
              </div>
            </div>

            <div className="table-wrap history-table-wrap">
              <table className="history-log-table">
                <thead>
                  <tr>
                    <th>시간</th>
                    <th>상태</th>
                    <th>구간</th>
                    <th>위험도</th>
                    <th>파일명</th>
                  </tr>
                </thead>

                <tbody>
                  {history.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="empty">
                        최근 생체신호 로그가 없습니다.
                      </td>
                    </tr>
                  ) : (
                    history.slice(0, 10).map((item, index) => (
                      <tr key={`${item.time}-${index}`}>
                        <td>{item.time}</td>
                        <td>
                          <span
                            className={`badge ${getStatusClass(item.status)}`}
                          >
                            {statusText(item.status, item.state)}
                          </span>
                        </td>
                        <td>
                          {item.total_segments ?? item.segments?.length ?? "-"}
                        </td>
                        <td>{item.risk_score ?? 0}점</td>
                        <td className="file-cell">{item.source || "-"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </article>
        </aside>
      </main>

      <article className="card behavior-full-card">
        <div className="section-title-row">
          <div>
            <h2>생체신호 패턴 및 원인 분석</h2>
            <p className="desc">
              원본 VitalSignal CSV는 1초 단위 feature로 자동 변환하고, 전처리
              CSV는 그대로 모델 입력값으로 사용합니다.
            </p>
          </div>

          <span className="behavior-chip">
            {isPlaying
              ? "Realtime"
              : result?.mode || "autoencoder_realtime_window"}
          </span>
        </div>

        <div className="behavior-summary">
          <strong>
            {latestSegment
              ? `${latestSegment.segment_index}번 구간은 ${statusText(
                  latestSegment.status,
                  latestSegment.state,
                )} 상태입니다.`
              : "CSV를 업로드하고 실시간 시작을 누르면 구간별 분석이 표시됩니다."}
          </strong>

          <p>
            {latestSegment
              ? latestSegment.message
              : "원본 VitalSignal CSV와 전처리 feature CSV를 모두 지원하며 1초 단위로 하나씩 분석합니다."}
          </p>
        </div>

        <div className="behavior-grid">
          <div>
            <span>분석 방식</span>
            <strong>
              {inputModeText(
                result?.input_mode || simulation?.input_mode,
                result?.input_description || simulation?.input_description,
              )}
            </strong>
          </div>

          <div>
            <span>구간 기준</span>
            <strong>{health?.window_seconds ?? "-"}초 단위</strong>
          </div>

          <div>
            <span>보호자 알림 기준</span>
            <strong>
              위험 상태가 지속될 경우 보호자 확인이 필요한 이벤트로 분류합니다.
            </strong>
          </div>
        </div>
      </article>
    </div>
  );
}

export default VitalDashboard;
