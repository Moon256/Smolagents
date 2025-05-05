from smolagents import tool

# Tool 1: Lấy tiêu đề + link bài viết mới từ vnexpress.net
@tool
def fetch_latest_news_titles_and_urls(url: str) -> list[tuple[str, str]]:
    """
    Trích xuất tiêu đề và URL của các bài viết mới nhất từ VnExpress.
    Args:
        url (str): URL trang chủ VnExpress (ví dụ: https://vnexpress.net)
    Returns:
        list[tuple[str, str]]: Danh sách [(tiêu đề, URL)]
    """
    import requests
    from bs4 import BeautifulSoup

    article_urls = []
    article_titles = []
    navigation_urls = []

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    navigation_bar = soup.find("nav", class_="main-nav")
    if navigation_bar:
        for header in navigation_bar.ul.find_all("li")[2:7]:  # Lấy mục chính
            navigation_urls.append(url + header.a["href"])

    for section_url in navigation_urls:
        response = requests.get(section_url)
        section_soup = BeautifulSoup(response.text, "html.parser")
        for article in section_soup.find_all("article"):
            title_tag = article.find("h3", class_="title-news")
            if title_tag:
                title = title_tag.text.strip()
                article_url = article.find("a")["href"]
                article_titles.append(title)
                article_urls.append(article_url)

    return list(zip(article_titles, article_urls))

# Tool 2: Trích nội dung đầy đủ từ một bài báo
@tool
def extract_news_article_content(url: str) -> str:
    """
    Trích xuất nội dung từ URL bài báo.
    Args:
        url (str): Đường dẫn bài báo
    Returns:
        str: Nội dung văn bản bài viết
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = ""
    for paragraph in soup.find_all("p"):
        content += paragraph.get_text().strip() + " "
    return content

# Tool 3: Tóm tắt nội dung tiếng Việt bằng mô hình vit5-base
@tool
def summarize_news(text: str) -> str:
    """
    Tóm tắt nội dung tiếng Việt sử dụng mô hình vit5-base của VietAI.
    Args:
        text (str): Văn bản cần tóm tắt
    Returns:
        str: Bản tóm tắt
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "VietAI/vit5-base-vietnews-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)

    formatted_text = "vietnews: " + text + " </s>"
    encoding = tokenizer(formatted_text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Tool 4: Phân loại xem bài báo có liên quan đến chủ đề không (zero-shot)
@tool
def classify_topic(text: str, topic: str) -> bool:
    """
    Phân loại một đoạn văn bản có liên quan đến chủ đề cụ thể hay không.
    Args:
        text (str): Đoạn văn bản
        topic (str): Chủ đề cần kiểm tra (ví dụ: "trí tuệ nhân tạo")
    Returns:
        bool: True nếu liên quan, False nếu không
    """
    from transformers import pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline(
        "zero-shot-classification",
        model="vicgalle/xlm-roberta-large-xnli-anli",
        device=0 if device == "cuda" else -1,
        trust_remote_code=True,
    )

    candidate_labels = [topic, f"không liên quan {topic}"]
    result = classifier(text, candidate_labels)
    return result["labels"][0] == topic
