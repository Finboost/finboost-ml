# Wajib lower case

FINANCIAL_KEYWORDS = [
    "finansial", "keuangan", "saham", "investasi", "asuransi", "hutang", "pensiun",
    "pinjaman", "kredit", "pajak", "dana", "uang", "transaksi", "ulti nolan", "bitcoin", "ethereum",
    "mata uang", "obligasi", "ekuitas", "portofolio", "dividen", "pinjam", "btc", "umr",
    "modal", "manajemen keuangan", "inflasi", "likuiditas", "aset", "rasio", "money",
    "liabilitas", "arisan", "anggaran", "biaya", "pendapatan", "menabung", "menambang",
    "manajemen risiko", "pendanaan", "reksadana", "laba", "neraca", "cripto", "tukar",
    "kas", "investor", "debitur", "kreditur", "suku bunga", "nambang", "budget", 
    "hipotek", "tabungan", "jangka", "risiko", "operasional", "economic", "inves",
    "rekening", "sistem","pembayaran", "perbankan", "biaya transaksi", "order", "fund",
    "komoditas", "derivatif", "keuangan pribadi", "keuangan korporasi", "spekulasi",
    "analisis", "fundamental","teknis", "trading", "manajemen portofolio", "pinjol",
    "arbitrase", "diversifikasi", "strategi investasi", "margin", "ekonomi", 
    "perdagangan", "pasar", "kebijakan moneter", "kebijakan fiskal", "intrinsik",
    "harga saham", "return on investment", "dampak ekonomi", "efisiensi pasar", "indikator ekonomi",
    "siklus ekonomi", "sistem perpajakan", "kebijakan keuangan", "pialang", "swot",
    "klaim", "uang tunai", "uang elektronik", "uang kertas", "sell", "buy", "pip",
    "uang logam", "uang digital", "uang kripto", "manajemen hutang", "manajemen piutang",
    "manajemen likuiditas", "manajemen risiko pasar", "manajemen risiko kredit", "utang",
    "hutang", "asuransi jiwa", "leverage", "roi", "ebitda", "pundi-pundi","bibit",
    "capital gain", "revenue", "expense", "balance sheet", "income statement", "financial",
    "cash flow", "profit", "loss", "earnings", "depreciation", "bayar", "beli",
    "amortization", "liquidity", "solvency", "ratio", "interest rate",
    "bond", "stock", "equity", "dividend", "investment", "bank", "finboost", "nilai"
    "planning", "retirement planning", "estate planning", "tax", "budgeting",
    "analysis", "modeling", "risk management", "credit risk", "market risk",
    "operational risk", "insurance", "health insurance", "life insurance", "auto insurance",
    "property insurance", "travel insurance", "insurance claim", "currency", "cryptocurrency",
    "digital currency", "fiat currency", "commodity", "derivative", "futures",
    "options", "hedging", "speculation", "arbitrage", "market", "dolar", "rupiah",
    "capital market", "money market", "foreign exchange", "exchange rate", "monetary policy",
    "fiscal policy", "economic policy", "regulation", "stability", "crisis",
    "banking", "central", "commercial", "retail", "payment", "nab", 
    "online", "mobile banking", "banking system", "payment system", "technology",
    "fintech", "blockchain", "decentralized", "defi", "initial coin offering",
    "ico", "smart contract", "wealth management", "asset management", "portfolio management",
    "advisor", "investment advisor", "stockbroker", "broker", "trader",
    "investor", "shareholder", "stakeholder", "venture capital", "private equity",
    "angel investor", "crowdfunding", "fundraising", "literacy", "education",
    "personal", "corporate", "public", "international", "development",
    "microfinance", "green", "sustainable", "socially responsible investing", "sri",
    "environmental", "social","governance", "esg", "impact investing", "inclusion", "innovation",
    "stability", "crisis", "credit crunch", "recession", "depression", "index",
    "economic growth", "economic development", "gross domestic product", "gdp", "gross national product",
    "gnp", "inflation rate", "deflation", "stagflation", "hyperinflation", "receh",
    "unemployment rate", "labor market", "consumer", "price", "cpi", "producer",
    "ppi", "leading economic indicators", "lagging economic indicators", "coincident economic indicators", "business",
    "expansion", "peak", "contraction", "trough", "recovery", "rugi", "financial",
    "economic indicator", "forecast", "economic forecast", "projection", "estimate",
    "budget forecast", "statement analysis", "ratio analysis", "trend analysis", "horizontal analysis",
    "vertical analysis", "comparative analysis", "health", "performance", "position",
    "stability", "security", "independence", "freedom", "success",
    "goal", "plan", "objective", "strategy", "management", "inkubasi",
    "control", "decision", "policy", "governance", "leadership",
    "responsibility", "accountability", "transparency", "integrity", "ethics",
    "culture", "behavior", "attitude", "psychology", "behavioral", "kaya",
    "neurofinance", "emotional", "finance", "empowerment", "resilience", "wellbeing",
    "pengelolaan dana", "pengeluaran", "pasif", "pendapatan aktif", "rasio keuangan",
    "surplus", "defisit", "rekening koran", "ekonomi mikro", "ekonomi makro", "penghasilan",
    "kontrak berjangka", "spekulasi pasar", "pelaporan keuangan", "laporan tahunan", "laporan triwulan",
    "angsuran", "manajemen risiko asuransi", "daya beli", "keseimbangan neraca", "penghindaran pajak",
    "optimasi pajak", "konsolidasi utang", "refinancing", "saham preferen", "keuangan publik",
    "rasio utang terhadap ekuitas", "rasio lancar", "penilaian aset", "pencucian uang", "transfer dana",
    "analisis pasar", "kesehatan keuangan", "kinerja keuangan", "penerimaan kas", "gaji",
    "duit", "ngutang", "ngasih pinjem", "bayar utang", "gaji", "credit", "cfa", "cfp",
    "duit bulanan", "bayaran", "tip", "bon", "beli barang", "kpr", "cicil", "sukuk",
    "cicilan", "tagihan", "pengeluaran harian", "dompet", "rekening bank",
    "transfer", "tarik tunai", "setor tunai", "atm", "bunga pinjaman", "pajak",
    "tabung", "jajan", "belanja", "pinjem duit", "kartu kredit", "lunas", "terima",
    "kartu debit", "asuransi mobil", "asuransi rumah", "asuransi kesehatan", "makasih",
    "dana darurat", "penghasilan", "pemasukan", "pengeluaran", "pendapatan tambahan", 
    "uang lembur", "uang saku", "budget harian", "saldo", "hutang kartu kredit",
    "angsuran kredit", "hutang rumah", "hutang mobil", "gaji pokok", "uang makan",
    "bonus", "insentif", "kas kecil", "uang pensiun", "biaya hidup", "obligasi", "emas", "cryptocurrency", 
    "mengatasinya", "memperbaikinya", "mengelolanya", "mengelola", "memberi", "penipuan", "penipu", "ditipu"
]
