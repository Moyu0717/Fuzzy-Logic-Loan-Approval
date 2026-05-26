# ================================================================
# Loan Approval Intelligence — Streamlit + Gemini AI Agent
# ----------------------------------------------------------------
# The fuzzy engine (FIS + GA) lives in fuzzy_engine.py — unchanged.
# This file: Streamlit UI + a Gemini tool-calling agent whose loop is
#   observe -> collect -> reason -> call tools -> explain -> act
# The fuzzy engine is exposed to the agent as a TOOL.
# ================================================================

import os
import json
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Loan Approval AI Agent", page_icon="🏦", layout="wide")

# ----------------------------------------------------------------
# 1. Load the fuzzy engine ONCE (GA runs on first load, then cached)
# ----------------------------------------------------------------
@st.cache_resource(show_spinner="Running GA optimisation (first load only)…")
def load_engine():
    import fuzzy_engine as fe
    return fe

fe = load_engine()


# ----------------------------------------------------------------
# 2. TOOLS  — wrap the existing engine
# ----------------------------------------------------------------
def compute_risk_detailed(income, credit, ratio, default, emp):
    """Run the fuzzy engine + return score, decision, and membership degrees."""
    default_val = 1 if str(default).lower() in ("yes", "y", "1", "true") else 0
    fe.risk_sim.input["income"]  = float(income)
    fe.risk_sim.input["credit"]  = float(credit)
    fe.risk_sim.input["ratio"]   = float(ratio)
    fe.risk_sim.input["default"] = default_val
    fe.risk_sim.input["emp_exp"] = float(emp)
    fe.risk_sim.compute()
    score = float(fe.risk_sim.output["risk"])
    decision = "APPROVE" if score < 40 else ("REVIEW" if score < 70 else "REJECT")
    md = fe.membership_degree
    deg = {
        "income_high": round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["high"]), 2),
        "income_low":  round(md(float(income), fe.best_mfs["inc_u"], fe.best_mfs["inc"]["low"]), 2),
        "ratio_high":  round(md(float(ratio),  fe.best_mfs["rat_u"], fe.best_mfs["rat"]["high"]), 2),
        "ratio_low":   round(md(float(ratio),  fe.best_mfs["rat_u"], fe.best_mfs["rat"]["low"]), 2),
        "emp_experienced": round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["experienced"]), 2),
        "emp_junior":      round(md(float(emp), fe.best_mfs["emp_u"], fe.best_mfs["emp"]["junior"]), 2),
        "credit_risk": round(fe.ctos_risk(float(credit)), 2),
        "defaulted":   default_val,
    }
    return {"risk_score": round(score, 1), "decision": decision,
            "membership_degrees": deg,
            "rule_weights": {"ratio": fe.W_RATIO, "credit": fe.W_CREDIT,
                             "income": fe.W_INCOME, "default": fe.W_DEFAULT, "emp": fe.W_EMP}}


def lookup_ctos_category(credit):
    c = float(credit)
    cat = ("Very Poor" if c <= 449 else "Poor" if c <= 549 else
           "Fair" if c <= 649 else "Good" if c <= 749 else "Excellent")
    return {"credit_score": c, "category": cat, "risk_value": round(fe.ctos_risk(c), 2)}


# ----------------------------------------------------------------
# 3. Gemini tool (function) declarations
# ----------------------------------------------------------------
TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculate_loan_risk",
        description=("Run the GA-optimised fuzzy logic engine on one applicant. "
                     "Returns risk score 0-100, decision (APPROVE/REVIEW/REJECT) and the "
                     "membership degrees of each variable. Call when you have all 5 variables. "
                     "May be called MULTIPLE times for what-if scenarios."),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "income": types.Schema(type=types.Type.NUMBER, description="Annual income RM"),
                "credit": types.Schema(type=types.Type.NUMBER, description="CTOS score 300-850"),
                "ratio":  types.Schema(type=types.Type.NUMBER, description="Loan as fraction of income 0-1"),
                "default":types.Schema(type=types.Type.STRING, description="'Yes' or 'No'"),
                "emp":    types.Schema(type=types.Type.NUMBER, description="Employment years"),
            },
            required=["income", "credit", "ratio", "default", "emp"],
        ),
    ),
    types.FunctionDeclaration(
        name="lookup_ctos_category",
        description="Look up the CTOS credit category for a credit score, as extra evidence.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"credit": types.Schema(type=types.Type.NUMBER, description="CTOS score")},
            required=["credit"],
        ),
    ),
])

def execute_tool(name, args):
    if name == "calculate_loan_risk":
        return compute_risk_detailed(args["income"], args["credit"], args["ratio"],
                                     args["default"], args["emp"])
    if name == "lookup_ctos_category":
        return lookup_ctos_category(args["credit"])
    return {"error": f"unknown tool {name}"}


SYSTEM = """You are a Malaysian loan officer AI agent. You assess loan applications using a
fuzzy-logic risk engine that you call as a tool.

Follow this loop and briefly show each step:
1. OBSERVE  – note which of the 5 variables you have: income (RM/year), credit (300-850),
   loan ratio (0-1), previous default (Yes/No), employment experience (years).
2. COLLECT  – if a variable is missing, ask ONE concise question. You may call
   lookup_ctos_category to add credit evidence.
3. REASON   – when you have all 5, decide to call the risk engine.
4. CALL TOOL– call calculate_loan_risk. If the decision is REJECT or REVIEW, try a what-if:
   call again with a small realistic change (lower the loan ratio, +1 year experience) to
   see if it would flip to a better decision.
5. EXPLAIN  – explain using the membership_degrees the tool returned (do NOT invent reasons).
6. ACT      – give the final decision and, if not APPROVE, 2-3 concrete improvement steps.

Reply in the user's language (Chinese or English). Be concise."""


# ----------------------------------------------------------------
# 4. Agent loop (manual function-calling so we can show every step)
# ----------------------------------------------------------------
def get_client():
    key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        return None
    return genai.Client(api_key=key)


def run_agent(client, history, user_msg, render):
    """history = list[types.Content]; render(kind, payload) updates the UI live."""
    contents = history + [types.Content(role="user", parts=[types.Part(text=user_msg)])]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[TOOLS],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    for _ in range(8):  # safety cap
        resp = client.models.generate_content(
            model="gemini-2.0-flash", contents=contents, config=cfg)
        cand = resp.candidates[0]
        parts = cand.content.parts or []

        # surface any text
        for p in parts:
            if getattr(p, "text", None):
                render("text", p.text.strip())

        fcs = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if fcs:
            contents.append(cand.content)  # model's turn (with the calls)
            resp_parts = []
            for fc in fcs:
                args = dict(fc.args)
                render("tool", (fc.name, args))
                result = execute_tool(fc.name, args)
                render("result", (fc.name, result))
                resp_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=resp_parts))
            continue
        else:
            contents.append(cand.content)
            break

    return contents


# ----------------------------------------------------------------
# 5. UI
# ----------------------------------------------------------------
st.title("🏦 Loan Approval Intelligence")
st.caption("GA-optimised Fuzzy Logic engine + Gemini AI Agent · the engine is a *tool* the agent calls")

ga = fe.ga_result
st.success(f"GA Fitness: {ga['init_fitness']:.3f} → {ga['best_fitness']:.3f} "
           f"({ga['improvement_pct']:.1f}% improvement)")

left, right = st.columns([1, 1.3])

# ---- Left: manual fuzzy evaluation (the original system) ----
with left:
    st.subheader("Manual evaluation")
    with st.form("manual"):
        c1, c2 = st.columns(2)
        income = c1.number_input("Annual income (RM)", 0, 1_000_000, 65000, step=1000)
        credit = c2.number_input("Credit score (CTOS)", 300, 850, 720)
        ratio  = c1.number_input("Loan % of income", 0.0, 1.0, 0.18, step=0.01)
        emp    = c2.number_input("Employment years", 0.0, 50.0, 4.0, step=0.5)
        default = st.selectbox("Previous default", ["No", "Yes"])
        go = st.form_submit_button("Evaluate Risk", use_container_width=True)
    if go:
        r = compute_risk_detailed(income, credit, ratio, default, emp)
        color = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}[r["decision"]]
        st.markdown(f"### Risk: {r['risk_score']}/100 — :{color}[{r['decision']}]")
        st.json(r["membership_degrees"])
    st.caption("CTOS: 300-449 Very Poor · 450-549 Poor · 550-649 Fair · 650-749 Good · 750-850 Excellent")

# ---- Right: AI Agent chat ----
with right:
    st.subheader("🤖 AI Agent")
    st.caption('Try: "月薪5000，CTOS 680，借收入的两成，没违约，做了3年"  ·  '
               '"income 80k, credit 600, ratio 0.45, no default, 1 year"')

    client = get_client()
    if client is None:
        st.warning("No GEMINI_API_KEY found. Add it in **Settings → Secrets** "
                   "(deployed) or as an environment variable (local). "
                   "Get a free key at aistudio.google.com.")

    if "gem_history" not in st.session_state:
        st.session_state.gem_history = []     # list[types.Content]
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []        # list[(role, text)] for display

    # render past chat
    for role, text in st.session_state.chat_log:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("Describe an applicant…", disabled=(client is None))
    if prompt:
        st.session_state.chat_log.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            box = st.container()
            buf = {"text": []}

            def render(kind, payload):
                if kind == "text":
                    buf["text"].append(payload)
                    box.markdown(payload)
                elif kind == "tool":
                    name, args = payload
                    box.markdown(f"🔧 **calling** `{name}` → `{json.dumps(args, ensure_ascii=False)}`")
                elif kind == "result":
                    name, res = payload
                    if name == "calculate_loan_risk":
                        box.markdown(f"↳ score **{res['risk_score']}** → **{res['decision']}**")
                        box.json(res["membership_degrees"])
                    else:
                        box.json(res)

            with st.spinner("Agent thinking…"):
                st.session_state.gem_history = run_agent(
                    client, st.session_state.gem_history, prompt, render)

            st.session_state.chat_log.append(("assistant", "\n\n".join(buf["text"]) or "_(done)_"))
