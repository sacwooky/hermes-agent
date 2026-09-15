def data_url_to_part(url: str) -> dict[str, Any]:
    header, b64 = url.split(",", 1)
    mime = "image/png"
    if header.startswith("data:"):
        mime = header[5:].split(";", 1)[0] or mime
    return {"inlineData": {"mimeType": mime, "data": b64}}

def content_to_parts(content: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if content is None:
        return parts
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
        return parts
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ == "text":
                text = item.get("text") or ""
                if text:
                    parts.append({"text": text})
            elif typ == "image_url":
                image = item.get("image_url") or {}
                url = image.get("url") if isinstance(image, dict) else str(image)
                if not url:
                    continue
                if url.startswith("data:"):
                    parts.append(data_url_to_part(url))
                elif url.startswith("http://") or url.startswith("https://"):
                    parts.append(fetch_image_to_part(url))
        return parts
    parts.append({"text": str(content)})
    return parts
