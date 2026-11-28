"""Conservative, offline Russian/English/Polish search phrase parser.

Calendar dates use the supplied local clock; emitted dates are calendar labels.
The existing search engine interprets > as strictly after the entire day, so
closed calendar intervals use the preceding day as their lower bound.
"""
import calendar
from datetime import datetime, timedelta
import re


_TYPES = (
    ('pdf', r'pdf(?:-ки|-ка|-ок|s)?|пдф(?:-ки|-ка|-ок)?'),
    ('doc,docx,odt,rtf,txt', r'документ\w*|ворд\w*|word|текст(?:ы|а|ов|овый|овые|овых)?|documents?|texts?|text\s+files?|dokument\w*|tekst\w*'),
    ('xls,xlsx,ods,csv', r'таблиц\w*|эксел\w*|excel|spreadsheets?|arkusz\w*|tabel\w*'),
    ('ppt,pptx,odp', r'презентаци\w*|presentations?|prezentacj\w*'),
    ('jpg,jpeg,png,tif,tiff,bmp,webp,heic', r'фото\w*|картин\w*|изображени\w*|скан\w*|photos?|pictures?|images?|scans?|zdję\w*|obraz\w*|skan\w*'),
    ('mp4,mkv,avi,mov', r'видео|videos?|movies?|wideo|film\w*'),
    ('mp3,flac,wav,m4a', r'музык\w*|аудио|music|audio|songs?|muzyk\w*'),
    ('zip,rar,7z', r'архив\w*|archives?|archiw\w*'),
    ('eml,msg', r'письм\w*|почт\w*|e-?mails?|letters?|poczt\w*|wiadomoś\w*|maile?'),
)

_MONTHS = (
    r'январ\w*|january|stycz\w*', r'феврал\w*|february|lut(?:y|ego|ym)',
    r'март\w*|march|mar(?:zec|ca|cu)', r'апрел\w*|april|kwie(?:cień|tnia|tniu)',
    r'ма(?:й|я|е)|may|maj(?:a|u)?', r'июн\w*|june|czerw\w*',
    r'июл\w*|july|lip(?:iec|ca|cu)', r'август\w*|august|sierp\w*',
    r'сентябр\w*|september|wrze(?:sień|śnia|śniu)',
    r'октябр\w*|october|październik\w*',
    r'ноябр\w*|november|listopad\w*', r'декабр\w*|december|grud\w*',
)

_FOLDERS = (
    ('downloads', r'(?:в|из)\s+загрузк\w*|(?:in|from)\s+(?:the\s+)?downloads|(?:w|z)\s+(?:folderze\s+)?pobran\w*|downloads|pobrane'),
    ('desktop', r'на\s+рабочем\s+столе|(?:on|from)\s+(?:the\s+)?desktop|(?:na|z)\s+pulpi\w*|desktop|pulpit'),
    ('documents', r'(?:в|из)\s+документах|(?:in|from)\s+(?:the\s+)?documents|(?:w|z)\s+(?:folderze\s+)?dokumentach'),
    ('pictures', r'(?:в|из)\s+(?:картинках|изображениях)|(?:in|from)\s+(?:the\s+)?pictures|(?:w|z)\s+(?:folderze\s+)?(?:obrazach|obrazkach|zdjęciach)'),
    ('music', r'(?:в|из)\s+музыке|(?:in|from)\s+(?:the\s+)?music|(?:w|z)\s+(?:folderze\s+)?muzyce'),
    ('videos', r'(?:в|из)\s+видео|(?:in|from)\s+(?:the\s+)?videos|(?:w|z)\s+(?:folderze\s+)?(?:wideo|filmach)'),
)

_STOP = re.compile(
    r'(?<!\w)(?:за|в|во|на|из|с|от|по|файл(?:ы|ов|а)?|найди|найти|покажи|показать|'
    r'find|show|me|please|files?|in|on|from|the|of|for|'
    r'znajdź|znajdz|pokaż|pokaz|proszę|pliki?|plików|w|we|na|z|ze|za)(?!\w)', re.I)


def _shift_month(day, count):
    index = day.year * 12 + day.month - 1 + count
    year, month = divmod(index, 12)
    month += 1
    return day.replace(year=year, month=month,
                       day=min(day.day, calendar.monthrange(year, month)[1]))


def parse(text, now=None):
    """Return chips, residual text, engine query and whether any chip was found.

    Unsupported phrases stay in residual text. Name/content chips are plain
    search terms because the engine has no name: or content: operators.
    Quoting a payload protects it from interpretation as another condition.
    """
    today = (datetime.now() if now is None else now).date()
    work = text
    found = []

    def consume(pattern, handler):
        nonlocal work
        def replace(match):
            chips = handler(match)
            if not chips:
                return match.group(0)
            for kind, token in chips:
                found.append((match.start(), len(found), {
                    'kind': kind, 'label': match.group(0).strip(), 'token': token}))
            # Preserve offsets and create a boundary for unquoted payloads.
            return '\x00' * len(match.group(0))
        work = re.sub(r'(?<![\w:-])(?:' + pattern + r')(?![\w-])', replace, work, flags=re.I)

    # Explicit payloads must not become types ("в названии документ").
    payload = r'(?P<value>"[^"\n]+"|«[^»\n]+»|\'[^\'\n]+\'|[^\s\x00]+)'
    def value(match):
        return match.group('value').strip('"«»\'')
    consume(r'(?:без\s+слова|без|не|without(?:\s+(?:the\s+)?word)?|excluding|bez(?:\s+słowa)?|nie)\s+' + payload,
            lambda m: [('exclude', '-' + word) for word in value(m).split()])
    for kind, prefix in (
        ('name', r'в\s+названии|in\s+(?:the\s+)?(?:name|filename)|w\s+nazwie'),
        ('content', r'в\s+тексте|содержит|in\s+(?:the\s+)?text|contains|w\s+tekście|zawiera'),
    ):
        consume(r'(?:' + prefix + r')\s+' + payload,
                lambda m, k=kind: [(k, value(m))])

    for folder, pattern in _FOLDERS:
        consume(pattern, lambda m, f=folder: [('folder', 'folder:' + f)])

    def interval(start, end):
        return [('date', 'date:>' + (start - timedelta(days=1)).isoformat()),
                ('date', 'date:<' + end.isoformat())]

    for pattern, offset in ((r'сегодня|today|dzisiaj|dziś', 0),
                            (r'вчера|yesterday|wczoraj', -1)):
        consume(pattern, lambda m, n=offset: [('date', 'date:' + (today + timedelta(days=n)).isoformat())])
    for pattern, offset in (
        (r'(?:на\s+)?этой\s+неделе|this\s+week|(?:w\s+)?tym\s+tygodniu', 0),
        (r'(?:на\s+|за\s+)?прошл(?:ой\s+неделе|ую\s+неделю)|last\s+week|(?:w\s+)?(?:zeszłym|ubiegłym)\s+tygodniu', -1),
    ):
        start = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
        consume(pattern, lambda m, s=start: interval(s, s + timedelta(days=7)))
    for pattern, offset in (
        (r'(?:в\s+)?этом\s+месяце|(?:за\s+)?этот\s+месяц|this\s+month|(?:w\s+)?tym\s+miesiącu', 0),
        (r'(?:в\s+)?прошлом\s+месяце|(?:за\s+)?прошлый\s+месяц|last\s+month|(?:w\s+)?(?:zeszłym|ubiegłym)\s+miesiącu', -1),
    ):
        month = _shift_month(today.replace(day=1), offset)
        consume(pattern, lambda m, d=month: [('date', 'date:' + d.strftime('%Y-%m'))])
    for pattern, offset in (
        (r'(?:в\s+)?этом\s+году|(?:за\s+)?этот\s+год|this\s+year|(?:w\s+)?tym\s+roku', 0),
        (r'(?:в\s+)?прошлом\s+году|(?:за\s+)?прошлый\s+год|last\s+year|(?:w\s+)?(?:zeszłym|ubiegłym)\s+roku', -1),
    ):
        consume(pattern, lambda m, n=offset: [('date', 'date:' + str(today.year + n))])

    def relative(m):
        count = int(m.group('count'))
        unit = m.group('unit').lower()
        try:
            if unit.startswith(('меся', 'month', 'mies')):
                start = _shift_month(today, -count)
            else:
                start = today - timedelta(days=count * (7 if unit.startswith(('нед', 'week', 'tyg')) else 1))
        except (ValueError, OverflowError):
            return []
        return [('date', 'date:>' + start.isoformat())]
    consume(r'(?:за\s+последни[ех]|(?:in\s+)?(?:the\s+)?(?:last|past)|(?:za\s+)?ostatni\w*)\s+'
            r'(?P<count>\d{1,6})\s+(?P<unit>дн(?:ей|я)|день|недел\w*|месяц\w*|days?|weeks?|months?|dni|dzień|tygodni\w*|miesi\w*)', relative)

    for month, pattern in enumerate(_MONTHS, 1):
        consume(r'(?:(?:в|in|w|we)\s+)?(?:' + pattern + r')(?:\s+(?P<year>[12]\d{3})(?:\s+(?:года|году|roku))?)?',
                lambda m, n=month: [('date', 'date:%04d-%02d' % (int(m.group('year') or today.year), n))])
    consume(r'(?:за|в|in|w|z)\s+(?P<year>[12]\d{3})(?:\s+(?:год|года|году|roku))?',
            lambda m: [('date', 'date:' + m.group('year'))])

    units = {'кб': 'kb', 'kb': 'kb', 'kib': 'kb', 'мб': 'mb', 'mb': 'mb', 'mib': 'mb',
             'гб': 'gb', 'gb': 'gb', 'gib': 'gb', 'б': 'b', 'b': 'b'}
    def size(m):
        direction = '<' if m.group('op').lower() in ('меньше', 'less than', 'smaller than', 'under', 'poniżej', 'mniej niż', 'mniejsze niż') else '>'
        return [('size', 'size:' + direction + m.group('number').replace(',', '.') + units[m.group('unit').lower()])]
    consume(r'(?P<op>больше|меньше|more than|larger than|bigger than|less than|smaller than|over|under|powyżej|poniżej|więcej niż|mniej niż|większe niż|mniejsze niż)\s+'
            r'(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>кб|мб|гб|б|kib|mib|gib|kb|mb|gb|b)', size)
    consume(r'большие(?:\s+файлы)?|large(?:\s+files)?|big(?:\s+files)?|duże(?:\s+pliki)?',
            lambda m: [('size', 'size:>100mb')])
    consume(r'маленькие(?:\s+файлы)?|small(?:\s+files)?|małe(?:\s+pliki)?',
            lambda m: [('size', 'size:<100kb')])
    for extensions, pattern in _TYPES:
        consume(pattern, lambda m, e=extensions: [('type', 'ext:' + e)])

    if not found:
        return {'chips': [], 'text': text, 'query': text, 'understood': False}
    chips = [item[2] for item in sorted(found)]
    remainder = _STOP.sub(' ', work.replace('\x00', ' '))
    remainder = ' '.join(remainder.split()).strip(' ,;')
    query = ' '.join(filter(None, [remainder] + [chip['token'] for chip in chips]))
    return {'chips': chips, 'text': remainder, 'query': query, 'understood': True}
