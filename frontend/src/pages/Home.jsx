import { useNavigate } from "react-router-dom";
import "./Home.css";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <section className="home-hero">
        <div className="hero-left">
          <div className="service-badge">
            <span></span>
            Smart Care AI Monitoring Platform
          </div>

          <h1>
            독거노인 안전 관리를 위한
            <br />
            AI 협업 관제 플랫폼
          </h1>

          <p className="hero-desc">
            mmWave 센서와 생활·생체 센서 데이터를 기반으로 낙상 위험도와
            이상행동 상태를 분석합니다. 관제 담당자는 대시보드에서 위험 상황,
            보호자 알림 여부, 최근 기록을 확인하고 빠르게 대응할 수 있습니다.
          </p>

          <div className="hero-buttons">
            <button onClick={() => navigate("/fall")}>낙상 관제 열기</button>

            <button className="outline" onClick={() => navigate("/abnormal")}>
              이상행동 관제 열기
            </button>
          </div>

          <div className="hero-tags">
            <span>mmWave 낙상 감지</span>
            <span>이상행동 분석</span>
            <span>비영상 센서 기반</span>
            <span>보호자 알림</span>
          </div>
        </div>

        <div className="hero-right">
          <div className="monitor-card">
            <div className="monitor-header">
              <div>
                <p>CARE CONTROL</p>
                <h2>안전 관제 흐름</h2>
              </div>
              <span className="ready-badge">AI 분석</span>
            </div>

            <div className="monitor-flow">
              <FlowItem
                number="01"
                title="센서 데이터 수집"
                desc="mmWave CSV와 생활·생체 센서 데이터를 업로드합니다."
              />

              <FlowItem
                number="02"
                title="AI 행동 분석"
                desc="낙상 위험도와 위험, 주의, 외출, 식사, 수면 상태를 예측합니다."
              />

              <FlowItem
                number="03"
                title="위험 상황 판단"
                desc="Fall Alert 또는 위험/주의 상태인지 판단하고 기록 여부를 결정합니다."
              />

              <FlowItem
                number="04"
                title="관제 및 보호자 알림"
                desc="위험 상황은 관제 화면에 표시하고 보호자 확인 대상으로 관리합니다."
              />
            </div>
          </div>
        </div>
      </section>

      <section className="feature-section">
        <div className="section-title">
          <p>MAIN FEATURES</p>
          <h2>주요 기능</h2>
          <span>
            실제 구현한 낙상 예측, 이상행동 예측, 센서 정보 표시, 보호자 알림
            흐름을 중심으로 구성했습니다.
          </span>
        </div>

        <div className="feature-grid">
          <FeatureCard
            icon="01"
            title="낙상 감지"
            desc="mmWave CSV 데이터를 분석하여 낙상 위험도와 Fall Alert 여부를 판단합니다."
          />

          <FeatureCard
            icon="02"
            title="이상행동 분석"
            desc="실내 환경, 생체 신호, 활동 정보를 바탕으로 위험, 주의, 외출, 식사, 수면 상태를 예측합니다."
          />

          <FeatureCard
            icon="03"
            title="비영상 기반 관리"
            desc="카메라 영상 대신 센서 데이터를 활용하여 개인정보 부담을 줄입니다."
          />

          <FeatureCard
            icon="04"
            title="보호자 알림"
            desc="위험 상황이나 반복 주의 상황을 보호자 확인 필요 상태로 표시합니다."
          />
        </div>
      </section>

      <section className="scenario-section">
        <div className="scenario-card">
          <div className="scenario-title">
            <p>DETECTION SCENARIO</p>
            <h2>분석 대상 행동</h2>
          </div>

          <div className="scenario-grid">
            <ScenarioItem
              title="낙상 사고"
              desc="높이 변화, 속도 변화, 이후 움직임을 분석합니다."
            />

            <ScenarioItem
              title="위험 상태"
              desc="생체 신호나 활동 패턴이 위험 기준에 가까운 상태를 판단합니다."
            />

            <ScenarioItem
              title="주의 상태"
              desc="반복적으로 이상 징후가 감지되는 상태를 관찰합니다."
            />

            <ScenarioItem
              title="생활 상태"
              desc="외출, 식사, 수면, 정상 상태를 최신 예측 결과로 표시합니다."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function FlowItem({ number, title, desc }) {
  return (
    <div className="flow-item">
      <strong>{number}</strong>
      <div>
        <h3>{title}</h3>
        <p>{desc}</p>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="feature-card">
      <span>{icon}</span>
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  );
}

function ScenarioItem({ title, desc }) {
  return (
    <div className="scenario-item">
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  );
}

export default Home;
