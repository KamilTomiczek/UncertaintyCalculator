import uncertaintyHelper as uh
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Kalkulator Niepewności Pomiarowych", layout="wide")

st.title("📊 System Obliczania Niepewności Pomiarowych i Regresji Liniowej")
st.markdown("Aplikacja zgodna z wytycznymi konwencji **GUM** oraz podręcznika A. Zięby.")

# Tworzymy 3 zakładki funkcjonalne
tab1, tab2, tab3 = st.tabs([
    "📊 Tabela Pomiarowa i Regresja Liniowa", 
    "📈 Kalkulator Niepewności Typu A (Serie Pomiary)",
    "🧮 Kalkulator Niepewności Typu B (Przyrządy)"
])

# ==========================================
# ZAKŁADKA 1: TABELA I REGRESJA LINIOWA
# ==========================================
with tab1:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("1. Tabela Danych Pomiarowych")
        st.write("Wpisz punkty pomiarowe oraz ich niepewności $u(X)$ i $u(Y)$:")
        
        # Generator stałej niepewności typu B (upraszcza wpisywanie)
        with st.expander("🛠️ Automatyczny generator niepewności B dla całej tabeli"):
            st.caption("Jeśli Twój miernik ma stałą dokładność Δmax dla wszystkich punktów, wpisz ją poniżej:")
            delta_x_gen = st.number_input("Δmax dla osi X:", value=0.0, format="%.4f")
            delta_y_gen = st.number_input("Δmax dla osi Y:", value=0.0, format="%.4f")
            
            if st.button("Wypełnij kolumny u_X i u_Y ze wzoru Δmax / √3"):
                u_x_val = uh.uncertaintyTypeB(delta_x_gen) if delta_x_gen > 0 else 0.02
                u_y_val = uh.uncertaintyTypeB(delta_y_gen) if delta_y_gen > 0 else 0.05
                st.session_state['df_table'] = pd.DataFrame({
                    'X': [1.00, 2.00, 3.00, 4.00, 5.00],
                    'u_X': [u_x_val] * 5,
                    'Y': [2.10, 4.05, 6.15, 8.00, 10.12],
                    'u_Y': [u_y_val] * 5
                })
                st.rerun()

        # Inicjalizacja domyślnych danych
        if 'df_table' not in st.session_state:
            st.session_state['df_table'] = pd.DataFrame({
                'X': [1.00, 2.00, 3.00, 4.00, 5.00],
                'u_X': [0.02, 0.02, 0.02, 0.02, 0.02],
                'Y': [2.10, 4.05, 6.15, 8.00, 10.12],
                'u_Y': [0.05, 0.06, 0.08, 0.09, 0.11]
            })
        
        # Interaktywna tabela
        edited_df = st.data_editor(st.session_state['df_table'], num_rows="dynamic", use_container_width=True)
        
        st.header("2. Ustawienia Wykresu")
        label_x = st.text_input("Etykieta osi X (z jednostką):", value="Natężenie prądu I_S [mA]")
        label_y = st.text_input("Etykieta osi Y (z jednostką):", value="Napięcie Halla U_H [mV]")
        tytul_wykresu = st.text_input("Tytuł wykresu:", value="Badanie zależności napięcia Halla od prądu próbki")

    with col_right:
        st.header("3. Wyniki Regresji i Wykres")
        
        x_data = edited_df['X'].dropna().values
        u_x_data = edited_df['u_X'].dropna().values
        y_data = edited_df['Y'].dropna().values
        u_y_data = edited_df['u_Y'].dropna().values
        
        if len(x_data) >= 3 and len(x_data) == len(y_data) == len(u_x_data) == len(u_y_data):
            # Użycie funkcji linearRegression z uncertaintyHelper.py
            a, b, u_a, u_b = uh.linearRegression(x_data, y_data)
            
            # Wyświetlenie wyliczonych parametrów w formacie GUM (z użyciem format_result)
            st.subheader("Współczynniki prostej regresji ($y = ax + b$)")
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**Nachylenie (a):** {uh.format_result(a, u_a)}")
                st.caption(f"Niepewność u(a) = {u_a:.5f}")
            with c2:
                st.info(f"**Wyraz wolny (b):** {uh.format_result(b, u_b)}")
                st.caption(f"Niepewność u(b) = {u_b:.5f}")
                
            # Rysowanie wykresu
            fig, ax = plt.subplots(figsize=(9, 5.5))
            
            # Słupki niepewności
            ax.errorbar(
                x_data, y_data,
                xerr=u_x_data, yerr=u_y_data,
                fmt='o', color='blue', ecolor='red', elinewidth=1.2, capsize=4,
                label='Punkty pomiarowe z $u(x), u(y)$'
            )
            
            # Prosta regresji
            x_lin = np.linspace(min(x_data) - 0.1*(max(x_data)-min(x_data)), max(x_data) + 0.1*(max(x_data)-min(x_data)), 100)
            y_lin = a * x_lin + b
            ax.plot(x_lin, y_lin, 'r--', label=f'Prosta regresji: $y = {a:.3f}x + {b:.3f}$')
            
            ax.set_xlabel(label_x, fontsize=11)
            ax.set_ylabel(label_y, fontsize=11)
            ax.set_title(tytul_wykresu, fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)
            
            # Eksport wykresu
            st.subheader("📥 Eksport wykresu")
            col_png, col_pdf = st.columns(2)
            
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
            buf_png.seek(0)
            with col_png:
                st.download_button("🖼️ Pobierz jako PNG (300 DPI)", data=buf_png, file_name="wykres.png", mime="image/png")
                
            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            buf_pdf.seek(0)
            with col_pdf:
                st.download_button("📄 Pobierz jako PDF (wektorowy)", data=buf_pdf, file_name="wykres.pdf", mime="application/pdf")
        else:
            st.warning("⚠️ Wprowadź co najmniej 3 pełne punkty pomiarowe w tabeli.")


# ==========================================
# ZAKŁADKA 2: KALKULATOR NIEPEWNOŚCI TYPU A
# ==========================================
with tab2:
    st.header("📈 Obliczanie niepewności statystycznej $u_A$ dla serii pomiarów")
    st.markdown("Wklej serię wielokrotnych pomiarów tej samej wielkości fizycznej (oddzielone przecinkami lub spacjami):")
    
    dane_input = st.text_area("Pomiary wielokrotne:", value="10.2, 10.5, 10.1, 10.3, 10.4, 10.2")
    conf_level = st.selectbox("Poziom ufności (α):", [0.6826, 0.90, 0.95, 0.99], index=0)
    
    if dane_input:
        try:
            pomiary = [float(val.strip()) for val in dane_input.replace(',', ' ').split() if val.strip()]
            if len(pomiary) >= 2:
                srednia, u_a = uh.uncertaintyTypeA(pomiary, conf_level)
                t_student = uh.getStudentTFactor(len(pomiary), conf_level)
                
                st.subheader("Wyniki analizy statystycznej:")
                st.write(f"• Liczba pomiarów $N$: **{len(pomiary)}**")
                st.write(f"• Współczynnik Studenta-Fishera $t_{{\\alpha, N}}$: **{t_student:.3f}**")
                st.write(f"• Średnia arytmetyczna $\\bar{{x}}$: **{srednia:.5f}**")
                st.write(f"• Niepewność typu A $u_A(x)$: **{u_a:.5f}**")
                st.success(f"**Format końcowy GUM:** {uh.format_result(srednia, u_a)}")
            else:
                st.warning("⚠️ Wprowadź co najmniej 2 pomiary.")
        except ValueError:
            st.error("❌ Upewnij się, że wprowadzono poprawne liczby.")

# ==========================================
# ZAKŁADKA 3: KALKULATOR NIEPEWNOŚCI TYPU B
# ==========================================
with tab3:
    st.header("🧮 Obliczanie niepewności $u_B$ dla różnych przyrządów")
    st.markdown("Wybierz typ przyrządu, aby wyznaczyć maksymalną granicę błędu $\Delta_{max}$ oraz niepewność $u_B = \\frac{\Delta_{max}}{\\sqrt{3}}$.")
    
    typ_p = st.selectbox("Typ przyrządu pomiarowego:", [
        "Prosty mechaniczny (linijka, suwmiarka, mikrometr)",
        "Analogowy (wskazówkowy z klasą i zakresem)",
        "Cyfrowy (multimetr cyfrowy)",
        "Zliczający (np. licznik Geigera-Müllera)"
    ])
    
    if typ_p == "Prosty mechaniczny (linijka, suwmiarka, mikrometr)":
        dzialka = st.number_input("Najmniejsza działka skali (Δmax):", value=0.05, format="%.4f")
        u_b = uh.uncertaintyTypeB(dzialka)
        st.success(f"**Niepewność $u_B$:** {u_b:.5f}")

    elif typ_p == "Analogowy (wskazówkowy z klasą i zakresem)":
        klasa = st.number_input("Klasa dokładności miernika:", value=2.5, format="%.2f")
        zakres = st.number_input("Zakres pomiarowy miernika:", value=20.0, format="%.2f")
        delta_max = uh.deltaMaxAnalog(klasa, zakres)
        u_b = uh.uncertaintyTypeB(delta_max)
        st.info(f"Maksymalny błąd $\Delta_{{max}} = \\frac{{{klasa} \cdot {zakres}}}{{100}} = {delta_max:.4f}$")
        st.success(f"**Niepewność $u_B$:** {u_b:.5f}")

    elif typ_p == "Cyfrowy (multimetr cyfrowy)":
        wynik = st.number_input("Odczytany wynik pomiaru:", value=229.73, format="%.4f")
        klasa_p = st.number_input("Klasa przyrządu (a %):", value=0.5, format="%.2f")
        b_dgt = st.number_input("Liczba cyfr (b / dgt):", value=2, step=1)
        rozdz = st.number_input("Rozdzielczość (skok ostatniej cyfry):", value=0.01, format="%.4f")
        
        delta_max = uh.deltaMaxDigital(wynik, klasa_p, b_dgt, rozdz)
        u_b = uh.uncertaintyTypeB(delta_max)
        st.info(f"Maksymalny błąd $\Delta_{{max}} = \\frac{{{klasa_p}}}{{100}} \cdot {abs(wynik)} + {b_dgt} \cdot {rozdz} = {delta_max:.5f}$")
        st.success(f"**Niepewność $u_B$:** {u_b:.5f}")

    elif typ_p == "Zliczający (np. licznik Geigera-Müllera)":
        N_imp = st.number_input("Liczba zliczonych impulsów N:", value=100, step=1)
        u_b = uh.uncertaintyTypeBCounting(N_imp)
        st.success(f"**Niepewność $u_B(N) = \\sqrt{{N}}$:** {u_b:.5f}")
