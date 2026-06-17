# noshow-predictor / app.py
# Parte 9 — Aplicação Streamlit
# Execute com: streamlit run app.py
# Compatível com Windows e Linux

import json
import shap
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────
st.set_page_config(
    page_title='Previsão de No-Show',
    page_icon='🏥',
    layout='wide'
)

# ── CARREGAMENTO OU TREINAMENTO DO MODELO ────────────────────────
@st.cache_resource
def carregar_ou_treinar_modelo():
    caminho_modelo = Path('model') / 'model.pkl'
    caminho_feats  = Path('model') / 'feature_names.pkl'

    if caminho_modelo.exists() and caminho_feats.exists():
        modelo        = joblib.load(caminho_modelo)
        feature_names = joblib.load(caminho_feats)
        return modelo, feature_names

    caminho_treino = Path('data') / 'processed' / 'train.csv'
    if not caminho_treino.exists():
        st.error(
            'Dados de treino não encontrados em data/processed/train.csv. '
            'Execute primeiro o script notebooks/04_selecao_features.py '
            'ou notebooks/pipeline_completo.py.'
        )
        st.stop()

    treino        = pd.read_csv(caminho_treino)
    X_treino      = treino.drop(columns=['noshow'])
    y_treino      = treino['noshow']
    feature_names = X_treino.columns.tolist()

    neg = (y_treino == 0).sum()
    pos = (y_treino == 1).sum()

    modelo = xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = neg / pos,
        random_state     = 42,
        n_jobs           = -1,
    )
    modelo.fit(X_treino, y_treino)

    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo,        caminho_modelo)
    joblib.dump(feature_names, caminho_feats)

    return modelo, feature_names


@st.cache_data
def carregar_metricas():
    caminho = Path('assets') / 'metrics_report.json'
    if caminho.exists():
        with open(caminho, encoding='utf-8') as f:
            return json.load(f)
    return None


# Inicializar modelo
with st.spinner('Carregando o modelo... (pode levar alguns minutos no primeiro acesso)'):
    modelo, feature_names = carregar_ou_treinar_modelo()

metricas = carregar_metricas()

# ── CABEÇALHO ────────────────────────────────────────────────────
st.title('Previsão de No-Show em Consultas Médicas')
st.markdown(
    'Sistema de Machine Learning para prever a probabilidade de um '
    'paciente **não comparecer** à sua consulta médica agendada.'
)
st.divider()

# ── ABAS PRINCIPAIS ──────────────────────────────────────────────
aba_previsao, aba_metricas, aba_sobre = st.tabs([
    'Previsão', 'Métricas do Modelo', 'Sobre'
])

# ══════════════════════════════════════════════════════════════════
# ABA 1 — PREVISÃO
# ══════════════════════════════════════════════════════════════════
with aba_previsao:

    # ── SIDEBAR REESTRUTURADA ─────────────────────────────────────
    with st.sidebar:
        st.header('Dados do Paciente')
        st.caption('Insira as características clínicas e do agendamento para inferência.')
        st.divider()

        genero = st.selectbox(
            'Gênero', options=['F', 'M'],
            format_func=lambda x: 'Feminino' if x == 'F' else 'Masculino'
        )
        
        idade = st.slider('Idade (anos)', min_value=0, max_value=100, value=20)
        dias_antecedencia = st.slider('Dias de antecedência do agendamento', min_value=0, max_value=120, value=1)
        
        st.write("") 

        st.markdown('**Condições de saúde & sociais:**')
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            hipertensao   = st.checkbox('Hipertensão')
            diabetes      = st.checkbox('Diabetes')
            alcoolismo    = st.checkbox('Alcoolismo')
        with col_sb2:
            deficiencia   = st.checkbox('Deficiência')
            bolsa_familia = st.checkbox('Bolsa Família')

        st.write("") 
        sms_recebido = st.checkbox('Recebeu SMS de lembrete')

        st.divider()
        btn_prever = st.button('Executar Previsão', type='primary', use_container_width=True)

    # Área principal — resultado da previsão
    if btn_prever:
        entrada_raw = {
            'Age'              : int(idade),
            'dias_antecedencia': int(dias_antecedencia),
            'SMS_received'     : int(sms_recebido),
            'Hipertension'     : int(hipertensao),
            'Diabetes'         : int(diabetes),
            'Alcoholism'       : int(alcoolismo),
            'Handcap'          : int(deficiencia),
            'Scholarship'      : int(bolsa_familia),
            'Gender_F'         : int(genero == 'F'),
            'Gender_M'         : int(genero == 'M'),
        }

        entrada_df = pd.DataFrame([entrada_raw])
        entrada_df = entrada_df.reindex(columns=feature_names, fill_value=0)

        prob = modelo.predict_proba(entrada_df)[0][1]
        predicao = modelo.predict(entrada_df)[0]

        if prob < 0.30:
            status_risco = "BAIXO"
            cor_dinamica = "#28a745"
        elif prob < 0.55:
            status_risco = "MÉDIO"
            cor_dinamica = "#ffc107"
        else:
            status_risco = "ALTO"
            cor_dinamica = "#dc3545"
            
        decisao_texto = "Não Comparecerá" if predicao == 1 else "Comparecerá"

        # 📦 CARD DE RESULTADOS UNIFICADO
        with st.container(border=True):
            st.markdown("<p style='text-align: center; font-weight: bold; margin-bottom: 25px; font-size: 1.2rem;'> Diagnóstico do Agendamento</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='text-align: center; line-height: 1.2;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.8rem; color: #808495; margin-bottom: 12px;'>Probabilidade de No-Show</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.8rem; font-weight: bold; margin: 0; color: {cor_dinamica};'>{prob:.1%}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div style='text-align: center; line-height: 1.2;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.8rem; color: #808495; margin-bottom: 12px;'>Grau de Risco</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.8rem; font-weight: bold; margin: 0; color: {cor_dinamica};'>{status_risco}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div style='text-align: center; line-height: 1.2;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.8rem; color: #808495; margin-bottom: 12px;'>Decisão do Modelo</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.8rem; font-weight: bold; margin: 0; color: {cor_dinamica};'>{decisao_texto}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        # ── SEÇÃO EXPLICABILIDADE SHAP (GRÁFICO + RESUMO) ──────────────────────
        st.subheader('Por que o modelo fez essa previsão?')
        st.markdown('Passe o mouse sobre as barras para ver o impacto exato de cada fator no risco de No-Show.')

        explainer   = shap.TreeExplainer(modelo)
        shap_values = explainer(entrada_df)
        pesos = shap_values.values[0]
        
        mapa_nomes = {
            'Age': 'Idade do Paciente',
            'dias_antecedencia': 'Dias de Antecedência',
            'SMS_received': 'SMS de Lembrete',
            'Hipertension': 'Hipertensão',
            'Diabetes': 'Diabetes',
            'Scholarship': 'Bolsa Família',
            'Alcoholism': 'Alcoolismo',
            'Handcap': 'Deficiência',
            'Gender_F': 'Gênero: Feminino',
            'Gender_M': 'Gênero: Masculino'
        }

        dados_base = []
        for feat, peso in zip(feature_names, pesos):
            val = entrada_df[feat].values[0]
            nome_limpo = mapa_nomes.get(feat, feat)
            
            if feat in ['SMS_received', 'Hipertension', 'Diabetes', 'Scholarship', 'Alcoholism', 'Handcap', 'Gender_F', 'Gender_M']:
                val_txt = "Sim" if val == 1 else "Não"
            else:
                val_txt = str(val)
                
            dados_base.append({
                'Feature': f"{nome_limpo} ({val_txt})",
                'Impacto': peso,
                'Abs_Impacto': abs(peso)
            })

        df_grafico = pd.DataFrame(dados_base)
        df_grafico = df_grafico.sort_values(by='Abs_Impacto', ascending=True).tail(6)

        total_outros = len(feature_names) - 6
        impacto_outros = pesos.sum() - df_grafico['Impacto'].sum()
        if total_outros > 0:
            linha_outros = pd.DataFrame([{
                'Feature': f"Outras {total_outros} variáveis",
                'Impacto': impacto_outros,
                'Abs_Impacto': abs(impacto_outros)
            }])
            df_grafico = pd.concat([linha_outros, df_grafico], ignore_index=True)

        import plotly.graph_objects as go
        cores = ['#dc3545' if x > 0 else '#28a745' for x in df_grafico['Impacto']]

        fig = go.Figure(go.Bar(
            x=df_grafico['Impacto'],
            y=df_grafico['Feature'],
            orientation='h',
            marker=dict(color=cores, line=dict(width=0)),
            text=[f" {x:+.2f}" for x in df_grafico['Impacto']],
            textposition='inside',
            insidetextanchor='end',
            hovertemplate="<b>%{y}</b><br>Contribuição: %{x:+.3f}<extra></extra>"
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=10), height=350,
            font=dict(family="Inter, sans-serif", size=13, color="#FAFAFA"),
            xaxis=dict(title="Impacto na decisão (Valores SHAP)", showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=True, zerolinecolor='rgba(255, 255, 255, 0.3)', tickfont=dict(color='#808495')),
            yaxis=dict(showgrid=False, tickfont=dict(color='#FAFAFA', size=13))
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Painel Informativo Textual (Top 5)
        st.write("") 
        st.markdown("### Resumo dos Principais Fatores")
        st.markdown("Veja o detalhamento amigável das **5 características mais decisivas** para este paciente:")

        df_resumo = pd.DataFrame(dados_base)
        df_resumo = df_resumo.sort_values(by='Abs_Impacto', ascending=False).head(5)

        for _, row in df_resumo.iterrows():
            impacto = row['Impacto']
            feature_formatada = row['Feature']
            
            if impacto > 0:
                badge_html = "<span style='background-color: rgba(220, 53, 69, 0.15); color: #dc3545; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; margin-right: 10px; display: inline-block; min-width: 95px; text-align: center;'> AGRAVANTE</span>"
                texto_acao = f"aumentou o risco de falta em <span style='color: #dc3545; font-weight: bold;'>+{impacto:.3f}</span>"
            else:
                badge_html = "<span style='background-color: rgba(40, 167, 69, 0.15); color: #28a745; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85rem; margin-right: 10px; display: inline-block; min-width: 95px; text-align: center;'> ATENUANTE</span>"
                texto_acao = f"reduziu o risco de falta em <span style='color: #28a745; font-weight: bold;'>{impacto:.3f}</span>"

            st.markdown(
                f"""
                <div style='display: flex; align-items: center; background-color: rgba(255,255,255,0.02); padding: 12px; border-radius: 6px; margin-bottom: 8px; border-left: 4px solid { '#dc3545' if impacto > 0 else '#28a745' };'>
                    {badge_html}
                    <div style='color: #FAFAFA; font-size: 0.95rem; margin-left: 5px;'>
                        O fator <b>{feature_formatada}</b> — {texto_acao}.
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════
# ABA 2 — MÉTRICAS DO MODELO
# ══════════════════════════════════════════════════════════════════
with aba_metricas:
    
    # ── CONJUNTO DE MÉTRICAS (CARD SUPERIOR) ──
    st.markdown("Desempenho no Conjunto de Teste")
    
    with st.container(border=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        def mini_metric(coluna, titulo, valor):
            with coluna:
                st.markdown(
                    f"""
                    <div style='text-align: center; line-height: 1.2;'>
                        <p style='font-size: 0.8rem; color: #808495; margin-bottom: 8px;'>{titulo}</p>
                        <p style='font-size: 1.6rem; font-weight: bold; margin: 0; color: #FAFAFA;'>{valor}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
        
        # Valores estáticos do relatório alinhados com o layout premium
        mini_metric(col1, "Acurácia", "58.4%")
        mini_metric(col2, "Precisão", "35.7%")
        mini_metric(col3, "Recall", "57.1%")
        mini_metric(col4, "F1-score", "43.9%")
        mini_metric(col5, "AUC-ROC", "0.608")

    st.write("")

    # ── MATRIZ DE CONFUSÃO INTERATIVA COM PLOTLY ──
    st.markdown("### Matriz de Confusão")
    
    import plotly.figure_factory as ff
    z = [[3026, 2113], [879, 1171]]
    x = ['Previsto: Compareceu', 'Previsto: Não Compareceu']
    y = ['Real: Compareceu', 'Real: Não Compareceu']
    
    fig_cm = ff.create_annotated_heatmap(
        z[::-1], x=x, y=y[::-1], 
        colorscale=[[0, '#1e212b'], [0.5, '#2e3d52'], [1, '#1d63b8']],
        showscale=False
    )
    
    fig_cm.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260,
        margin=dict(l=40, r=40, t=20, b=20),
        font=dict(family="Inter, sans-serif", size=13, color="#FAFAFA")
    )
    
    for i in range(len(fig_cm.layout.annotations)):
        fig_cm.layout.annotations[i].font.size = 16
        fig_cm.layout.annotations[i].font.weight = 'bold'
        
    st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': False})
    
    st.write("")

    # ── ANÁLISE CLÍNICA / IMPACTO DE NEGÓCIO ──
    st.markdown("### Análise de Impacto Clínico")
    st.markdown("Interpretação prática do comportamento do modelo na rotina do posto ou clínica:")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(
            """
            <div style='background-color: rgba(40, 167, 69, 0.05); padding: 16px; border-radius: 8px; border-left: 4px solid #28a745; height: 100%;'>
                <p style='font-size: 0.85rem; color: #808495; margin: 0 0 4px 0;'>No-shows Reais Detectados</p>
                <h2 style='color: #28a745; margin: 0 0 8px 0; font-weight: bold;'>1.171</h2>
                <p style='font-size: 0.85rem; color: #FAFAFA; margin: 0;'>
                     <b>57.1%</b> das faltas foram antecipadas com sucesso, permitindo remanejamento de pauta.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with c2:
        st.markdown(
            """
            <div style='background-color: rgba(220, 53, 69, 0.05); padding: 16px; border-radius: 8px; border-left: 4px solid #dc3545; height: 100%;'>
                <p style='font-size: 0.85rem; color: #808495; margin: 0 0 4px 0;'>Não Detectados (Falsos Negativos)</p>
                <h2 style='color: #dc3545; margin: 0 0 8px 0; font-weight: bold;'>879</h2>
                <p style='font-size: 0.85rem; color: #FAFAFA; margin: 0;'>
                     Casos em que o modelo previu presença, mas o paciente <b>faltou</b>. Geram horários ociosos.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        
    with c3:
        st.markdown(
            """
            <div style='background-color: rgba(255, 193, 7, 0.05); padding: 16px; border-radius: 8px; border-left: 4px solid #ffc107; height: 100%;'>
                <p style='font-size: 0.85rem; color: #808495; margin: 0 0 4px 0;'>Alarmes Falsos (Falsos Positivos)</p>
                <h2 style='color: #ffc107; margin: 0 0 8px 0; font-weight: bold;'>2.113</h2>
                <p style='font-size: 0.85rem; color: #FAFAFA; margin: 0;'>
                     Pacientes que <b>iriam comparecer</b>, mas dispararam alerta. Evite ações muito agressivas com eles.
                </p>
            </div>
            """, unsafe_allow_html=True
        )

# ══════════════════════════════════════════════════════════════════
# ABA 3 — SOBRE
# ══════════════════════════════════════════════════════════════════
with aba_sobre:
    st.subheader('Sobre o Projeto')
    st.markdown('''
    Este projeto foi desenvolvido como material de mentoria em Machine Learning.
    O objetivo é demonstrar o pipeline completo de um projeto de ML aplicado à saúde pública:
    da exploração dos dados ao deploy de uma aplicação web interativa.

    **Dataset:** Medical Appointment No Shows — Kaggle
    (110.527 registros coletados em Vitória/ES, 2015-2016)

    **Algoritmo:** XGBoost (Extreme Gradient Boosting)

    **Interpretabilidade:** SHAP (SHapley Additive exPlanations)

    **Repositório:** https://github.com/MaiccGms8/noshow-predictor
    ''')