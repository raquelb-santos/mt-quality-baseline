"""Language normalization, ported from post-mt utils.js."""

from __future__ import annotations

from typing import Any

LANGUAGE_MAPPING: dict[str, str] = {
    "af-za": "Afrikaans (South Africa)",
    "sq-al": "Albanian (Albania)",
    "ar-sa": "Arabic (Saudi Arabia)",
    "hy-ma": "Armenian (Armenia)",
    "bn-bd": "Bengali (Bangladesh)",
    "bs-ba": "Bosnian (Bosnia and Herzegovina)",
    "ca-es": "Catalan (Spain)",
    "hr-hr": "Croatian (Croatia)",
    "cs-cz": "Czech (Czech Republic)",
    "da-dk": "Danish (Denmark)",
    "nl-be": "Dutch (Belgium)",
    "nl-nl": "Dutch (Netherlands)",
    "en-us": "English (United States)",
    "en-gb": "English (United Kingdom)",
    "eo": "Esperanto",
    "et-ee": "Estonian (Estonia)",
    "tl-ph": "Filipino (Philippines)",
    "fi-fi": "Finnish (Finland)",
    "fr-fr": "French (France)",
    "fr-be": "French (Belgium)",
    "fr-ca": "French (Canada)",
    "de-de": "German (Germany)",
    "el-gr": "Greek (Greece)",
    "gu-in": "Gujarati (India)",
    "hi-in": "Hindi (India)",
    "hu-hu": "Hungarian (Hungary)",
    "is-is": "Icelandic (Iceland)",
    "id-id": "Indonesian (Indonesia)",
    "it-it": "Italian (Italy)",
    "ja-jp": "Japanese (Japan)",
    "jw-id": "Javanese (Indonesia)",
    "ko-kr": "Korean (South Korea)",
    "la-la": "Latin",
    "mr-in": "Marathi (India)",
    "pl-pl": "Polish (Poland)",
    "pt-br": "Portuguese (Brazil)",
    "pt-pt": "Portuguese (Portugal)",
    "ro-ro": "Romanian (Romania)",
    "ru-ru": "Russian (Russia)",
    "sr-rs": "Serbian (Serbia)",
    "si-lk": "Sinhala (Sri Lanka)",
    "sk-sk": "Slovak (Slovakia)",
    "sl-si": "Slovenian (Slovenia)",
    "es-419": "Spanish (Latin America)",
    "es-es": "Spanish (Spain)",
    "es-mx": "Spanish (Mexico)",
    "es-ar": "Spanish (Argentina)",
    "sv-fi": "Swedish (Finland)",
    "sv-se": "Swedish (Sweden)",
    "ta-in": "Tamil (India)",
    "te-in": "Telugu (India)",
    "th-th": "Thai (Thailand)",
    "tr-tr": "Turkish (Turkey)",
    "uk-ua": "Ukrainian (Ukraine)",
    "vi-vn": "Vietnamese (Vietnam)",
    "cy-gb": "Welsh (United Kingdom)",
    "zh-cn": "Chinese (Simplified, China)",
    "zh-tw": "Chinese (Traditional, Taiwan)",
    "zh-hk": "Chinese (Traditional, Hong Kong)",
}

LANGUAGE_REVERSE_MAPPING: dict[str, str] = {name.lower(): code for code, name in LANGUAGE_MAPPING.items()}


def normalize(lang: str) -> tuple[str, str]:
    """Any input (code or display name) to (code, name); unknowns pass through."""
    value = str(lang).lower()
    if value in LANGUAGE_MAPPING:
        return value, LANGUAGE_MAPPING[value]
    return LANGUAGE_REVERSE_MAPPING.get(value, value), lang


def normalize_language(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters with clean_*_language_code/_name injected."""
    output = dict(parameters)
    for side in ("source", "target"):
        if parameters.get(f"{side}_language"):
            code, name = normalize(parameters[f"{side}_language"])
            output[f"clean_{side}_language_code"] = code
            output[f"clean_{side}_language_name"] = name.lower()
    return output


def language_variants(language: str) -> list[str]:
    """Full code and base code (en-us -> [en-us, en])."""
    return list(dict.fromkeys([language, str(language).split("-")[0]]))


# Word-boundary matching is meaningless for languages written without inter-word spacing.
UNSPACED_LANGUAGES = frozenset({"ja", "zh", "ko", "th", "lo", "km", "my"})


def is_unspaced_language(language_code: str | None) -> bool:
    return str(language_code or "").split("-")[0].lower() in UNSPACED_LANGUAGES
