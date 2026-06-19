function Home({ movePage }) {
  return (
    <div className="home-page">
      <section className="home-hero">
        <div className="home-hero-text">
          <p className="home-kicker">AI 기반 독거노인 안전 관리 서비스</p>
          <h1>
            스마트 돌봄을 위한 <br />
            독거노인 안전 관리 AI 협업 관제 플랫폼
          </h1>
          <p>
            mmWave 센서와 AI 분석 모델을 활용하여 독거노인의 낙상 사고를
            감지하고, 보호자와 관제 담당자에게 실시간 알림을 제공하는 협업
            관제 시스템입니다.
          </p>

          <div className="home-buttons">
            <button onClick={() => movePage("fall")}>낙상 관제 페이지 보기</button>
            <button className="outline" onClick={() => movePage("fall")}>
              CSV 예측 테스트
            </button>
          </div>
        </div>

        <div className="home-hero-card">
          <div className="home-elder">👵👴</div>
          <h2>Smart Care Monitoring</h2>
          <p>낙상 감지 · 실시간 알림 · 보호자 연계 · 협업 관제</p>

          <div className="home-mini-grid">
            <div>
              <span>관리 대상</span>
              <strong>24명</strong>
            </div>
            <div>
              <span>센서 연결</span>
              <strong>96%</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="home-section">
        <p className="home-kicker center">PROJECT OVERVIEW</p>
        <h2>프로젝트 핵심 기능</h2>

        <div className="home-feature-grid">
          <FeatureCard
            icon="🚨"
            title="노인 낙상 감지"
            desc="mmWave 센서 데이터를 CSV로 변환하고 AI 모델이 낙상 여부를 예측합니다."
          />
          <FeatureCard
            icon="📡"
            title="비영상 센서 기반"
            desc="카메라 영상 대신 mmWave 센서를 활용하여 개인정보 부담을 줄입니다."
          />
          <FeatureCard
            icon="🔔"
            title="실시간 알림"
            desc="Fall Alert 발생 시 보호자와 관제 담당자에게 알림을 전달합니다."
          />
          <FeatureCard
            icon="📊"
            title="협업 관제"
            desc="서버 상태, 예측 결과, 저장 로그를 하나의 대시보드에서 관리합니다."
          />
        </div>
      </section>

      <section className="home-info-grid">
        <div>
          <h3>개발 배경</h3>
          <p>
            독거노인은 낙상 사고가 발생해도 즉시 발견되기 어렵고, 보호자나
            돌봄 인력이 항상 옆에서 확인하기도 어렵습니다. 본 프로젝트는 이러한
            문제를 해결하기 위해 mmWave 센서 데이터를 활용해 낙상 위험을
            감지하고 관제 화면에서 빠르게 확인할 수 있도록 설계했습니다.
          </p>
        </div>

        <div>
          <h3>시스템 흐름</h3>
          <div className="home-flow">
            <span>mmWave 센서</span>
            <b>→</b>
            <span>CSV 변환</span>
            <b>→</b>
            <span>AI 예측</span>
            <b>→</b>
            <span>관제 알림</span>
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="home-feature-card">
      <div>{icon}</div>
      <h3>{title}</h3>
      <p>{desc}</p>
    </div>
  );
}

export default Home;
