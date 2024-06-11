# Wajib lower case

FINANCIAL_KEYWORDS = [
    "finansial", "keuangan", "saham", "investasi", "asuransi", "hutang", "pensiun",
    "pinjaman", "kredit", "pajak", "dana", "uang", "transaksi", "ulti nolan", "bitcoin", "ethereum",
    "mata uang", "obligasi", "ekuitas", "portofolio", "dividen", "pinjam", "btc",
    "modal", "manajemen keuangan", "inflasi", "likuiditas", "aset", "rasio", "money",
    "liabilitas", "arisan", "anggaran", "biaya", "pendapatan", "menabung", "menambang",
    "manajemen risiko", "pendanaan", "reksadana", "laba", "neraca", "cripto",
    "kas", "investor", "debitur", "kreditur", "nilai tukar", "suku bunga", "nambang",
    "hipotek", "tabungan", "investasi jangka panjang", "investasi jangka pendek",
    "dana pensiun", "rekening", "sistem pembayaran", "perbankan", "biaya transaksi",
    "komoditas", "derivatif", "keuangan pribadi", "keuangan korporasi", "spekulasi",
    "analisis fundamental", "analisis teknis", "trading", "manajemen portofolio", "laba rugi",
    "arbitrase", "diversifikasi", "strategi investasi", "margin", "ekonomi", 
    "perdagangan", "pasar modal", "kebijakan moneter", "kebijakan fiskal", "nilai intrinsik",
    "harga saham", "return on investment", "dampak ekonomi", "efisiensi pasar", "indikator ekonomi",
    "siklus ekonomi", "sistem perpajakan", "biaya modal", "kebijakan keuangan", "pialang",
    "spekulan", "asuransi kesehatan", "asuransi jiwa", "asuransi kendaraan", "asuransi properti",
    "asuransi perjalanan", "klaim asuransi", "uang tunai", "uang elektronik", "uang kertas",
    "uang logam", "uang digital", "uang kripto", "manajemen hutang", "manajemen piutang",
    "manajemen likuiditas", "manajemen risiko pasar", "manajemen risiko kredit", "manajemen risiko operasional", "utang",
    "hutang", "asuransi jiwa", "leverage", "roi", "ebitda", "pundi-pundi","bibit",
    "capital gain", "revenue", "expense", "balance sheet", "income statement",
    "cash flow", "profit", "loss", "earnings", "depreciation", "bayar", "beli",
    "amortization", "liquidity", "solvency", "financial ratio", "interest rate",
    "bond", "stock", "equity", "dividend", "investment", "bank", "finboost",
    "financial planning", "retirement planning", "estate planning", "tax planning", "budgeting",
    "financial analysis", "financial modeling", "risk management", "credit risk", "market risk",
    "operational risk", "insurance", "health insurance", "life insurance", "auto insurance",
    "property insurance", "travel insurance", "insurance claim", "currency", "cryptocurrency",
    "digital currency", "fiat currency", "commodity", "derivative", "futures",
    "options", "hedging", "speculation", "arbitrage", "financial market",
    "capital market", "money market", "foreign exchange", "exchange rate", "monetary policy",
    "fiscal policy", "economic policy", "financial regulation", "financial stability", "financial crisis",
    "banking", "central banking", "commercial banking", "investment banking", "retail banking",
    "online banking", "mobile banking", "banking system", "payment system", "financial technology",
    "fintech", "blockchain", "decentralized finance", "defi", "initial coin offering",
    "ico", "smart contract", "wealth management", "asset management", "portfolio management",
    "financial advisor", "investment advisor", "stockbroker", "broker", "trader",
    "investor", "shareholder", "stakeholder", "venture capital", "private equity",
    "angel investor", "crowdfunding", "fundraising", "financial literacy", "financial education",
    "personal finance", "corporate finance", "public finance", "international finance", "development finance",
    "microfinance", "green finance", "sustainable finance", "socially responsible investing", "sri",
    "environmental, social, and governance", "esg", "impact investing", "financial inclusion", "financial innovation",
    "financial stability", "financial crisis", "credit crunch", "recession", "depression",
    "economic growth", "economic development", "gross domestic product", "gdp", "gross national product",
    "gnp", "inflation rate", "deflation", "stagflation", "hyperinflation", "receh",
    "unemployment rate", "labor market", "consumer price index", "cpi", "producer price index",
    "ppi", "leading economic indicators", "lagging economic indicators", "coincident economic indicators", "business cycle",
    "expansion", "peak", "contraction", "trough", "recovery", "rugi", "financial",
    "economic indicator", "financial forecast", "economic forecast", "financial projection", "financial estimate",
    "budget forecast", "financial statement analysis", "ratio analysis", "trend analysis", "horizontal analysis",
    "vertical analysis", "comparative analysis", "financial health", "financial performance", "financial position",
    "financial stability", "financial security", "financial independence", "financial freedom", "financial success",
    "financial goal", "financial plan", "financial objective", "financial strategy", "financial management",
    "financial control", "financial decision", "financial policy", "financial governance", "financial leadership",
    "financial responsibility", "financial accountability", "financial transparency", "financial integrity", "financial ethics",
    "financial culture", "financial behavior", "financial attitude", "financial psychology", "behavioral finance",
    "neurofinance", "emotional finance", "financial empowerment", "financial resilience", "financial wellbeing",
    "pengelolaan dana", "pengeluaran", "pendapatan pasif", "pendapatan aktif", "rasio keuangan",
    "surplus", "defisit", "rekening koran", "ekonomi mikro", "ekonomi makro", "penghasilan",
    "kontrak berjangka", "spekulasi pasar", "pelaporan keuangan", "laporan tahunan", "laporan triwulan",
    "angsuran", "manajemen risiko asuransi", "daya beli", "keseimbangan neraca", "penghindaran pajak",
    "optimasi pajak", "konsolidasi utang", "refinancing", "saham preferen", "keuangan publik",
    "rasio utang terhadap ekuitas", "rasio lancar", "penilaian aset", "pencucian uang", "transfer dana",
    "arbitase mata uang", "analisis pasar", "kesehatan keuangan", "kinerja keuangan", "penerimaan kas", "gaji",
    "duit", "ngutang", "ngasih pinjem", "bayar utang", "gaji", "credit", "cfa", "cfp",
    "duit bulanan", "bayaran", "tip", "bon", "beli barang", "kpr", "cicil",
    "cicilan", "tagihan", "pengeluaran harian", "dompet", "rekening bank",
    "transfer", "tarik tunai", "setor tunai", "atm", "bunga pinjaman", "pajak",
    "tabung", "jajan", "belanja", "pinjem duit", "kartu kredit", "lunas", "terima",
    "kartu debit", "asuransi mobil", "asuransi rumah", "asuransi kesehatan", "makasih",
    "dana darurat", "penghasilan", "pemasukan", "pengeluaran", "pendapatan tambahan", 
    "uang lembur", "uang saku", "budget harian", "saldo", "hutang kartu kredit",
    "angsuran kredit", "hutang rumah", "hutang mobil", "gaji pokok", "uang makan",
    "bonus", "insentif", "kas kecil", "uang pensiun", "biaya hidup", "obligasi", "emas", "cryptocurrency", 
    "thanks","hallo", "terimakasih", "sorry", "maaf", "hi", "tes", "test", "p", "makasih", "kamu",
    "mengatasinya", "memperbaikinya", "mengelolanya", "mengelola", "memberi", "penipuan", "penipu", "ditipu"
]
