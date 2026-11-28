"""Image tags on top of the CLIP index: zero-shot labelling of every indexed picture with a user's tag list
(typed, imported from a file, or the built-in vocabulary), saved next to the image store, searchable by tag.

Tags are matched by meaning, so an English tag list works best with the English CLIP encoder.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import numpy as np

_TAG_LOCK = threading.RLock()

_OLD_BUILTIN = (
    "person,people,man,woman,child,baby,crowd,face,selfie,group photo,dog,cat,bird,horse,fish,insect,animal,cow,"
    "flower,tree,forest,mountain,beach,sea,river,lake,sky,sunset,snow,rain,clouds,desert,garden,landscape,city,street,"
    "building,house,bridge,road,car,truck,bus,train,airplane,boat,bicycle,motorcycle,food,pizza,cake,fruit,vegetables,"
    "coffee,drink,restaurant,kitchen,bedroom,room,furniture,table,computer,laptop,phone,screenshot,text document,"
    "receipt,invoice,passport,id card,contract,form,table with numbers,chart,diagram,map,poster,book,newspaper,"
    "handwriting,presentation slide,drawing,painting,cartoon,anime,comic,meme,logo,icon,game screenshot,video game,"
    "website,code,sports,football,concert,party,wedding,birthday,holiday,travel,night,indoor,outdoor,black and white,"
    "macro photo,portrait,architecture,art,toy,clothes,shoes,jewelry,tools,machine,medical image,x-ray,"
    "nature,water,fire,sunrise,fog,statue,museum,church,castle,market,school,office,garden flowers"
).split(",")

# Extra neutral vocabulary grouped by category (all English, no people-identification,
# no artists/brands; flat list is exposed via BUILTIN, groups keep it maintainable).
_EXTRA = {
    "people": (
        "teenager,senior,toddler,friends,family,couple,tourist,athlete,musician,"
        "chef,doctor,teacher,student,worker,driver,pedestrian,runner,swimmer,"
        "bride,groom,silhouette,back view,close up"
    ).split(","),
    "animals": (
        "puppy,kitten,rabbit,deer,bear,fox,wolf,lion,tiger,elephant,zebra,giraffe,"
        "monkey,sheep,goat,pig,chicken,duck,goose,owl,eagle,seagull,butterfly,bee,"
        "frog,turtle,dolphin,whale,shark,octopus,crab,squirrel,hamster,parrot,"
        "penguin,panda,koala,snail,swan,peacock,lizard,ant,mosquito,moth,beetle,"
        "caterpillar,dragonfly,starfish,jellyfish,seal,otter,kangaroo,camel,donkey,"
        "llama,hedgehog,mouse,rat,bat,reindeer,leopard,cheetah,rhino,hippo,crocodile,"
        "pelican,stork,flamingo,turkey,sparrow,pigeon,crow,raven,hawk,falcon,woodpecker"
    ).split(","),
    "plants": (
        "leaves,grass,meadow,bush,mushroom,cactus,palm tree,pine tree,blossom,rose,"
        "tulip,sunflower,daisy,orchid,lily,fern,ivy,moss,bamboo,reeds,sunflower field,"
        "vineyard,orchard,cornfield,wheat field"
    ).split(","),
    "nature_water_sky": (
        "waterfall,pond,stream,cliff,cave,valley,hill,island,ocean,wave,glacier,"
        "volcano,coral reef,rainbow,storm,lightning,stars,moon,aurora,dew,frost,"
        "mist,puddle,sand,dune,rock,pebble,spring,autumn leaves,winter,autumn,"
        "summer,ice,foggy morning,golden hour,blue sky,overcast,sunny,windy"
    ).split(","),
    "city_places": (
        "village,town,downtown,suburb,plaza,alley,park,playground,stadium,airport,"
        "station,harbor,library,hospital,hotel,cafe,shop,mall,supermarket,bakery,"
        "pharmacy,bank,theater,cinema,temple,mosque,tower,skyscraper,monument,fountain,"
        "wall,fence,gate,parking,tunnel,railway,crosswalk,traffic light,lamppost,"
        "bench,bridge at night,old town,ruins,palace,cottage,cabin,greenhouse,barn,"
        "windmill,lighthouse,dam,aqueduct,obelisk,pavilion,kiosk,bus stop,farm,ranch"
    ).split(","),
    "transport": (
        "van,taxi,ambulance,fire truck,police car,tractor,trailer,scooter,skateboard,"
        "subway,tram,ferry,ship,yacht,sailboat,canoe,kayak,helicopter,hot air balloon,"
        "rocket,forklift,crane,excavator,locomotive,convertible,pickup truck,minivan,"
        "golf cart,wheelchair,stroller"
    ).split(","),
    "food_drink": (
        "bread,sandwich,burger,pasta,salad,soup,rice,noodles,eggs,cheese,sausage,"
        "seafood,sushi,chocolate,ice cream,cookies,pancakes,breakfast,lunch,dinner,"
        "tea,juice,milk,picnic,grapes,apple,banana,orange,strawberry,watermelon,lemon,"
        "pear,peach,cherries,plum,avocado,tomato,cucumber,carrot,potato,onion,garlic,"
        "mushroom soup,cereal,honey,jam,butter,pastry,croissant,muffin,cupcake,pie,"
        "steak,grill,kebab,tacos,dumplings,curry,spices,market stall,food truck"
    ).split(","),
    "home_interior": (
        "living room,bathroom,shower,sofa,chair,bed,desk,lamp,mirror,curtain,carpet,"
        "shelf,wardrobe,stove,oven,fridge,microwave,washing machine,balcony,terrace,"
        "fireplace,staircase,window,door,roof,chimney,backyard,attic,basement,garage,"
        "hallway,porch,patio,armchair,coffee table,bookshelf,chandelier,candle,"
        "pillow,blanket,towel"
    ).split(","),
    "clothes_objects": (
        "jacket,coat,dress,shirt,pants,jeans,skirt,suit,hat,cap,scarf,gloves,boots,"
        "sneakers,sandals,sunglasses,watch,handbag,backpack,suitcase,umbrella,key,"
        "wallet,glasses,belt,tie,socks,sweater,hoodie,uniform,costume,pajamas,slippers,"
        "ring,necklace,bracelet,earrings,balloon,kite,teddy bear,doll,ball,board game,"
        "puzzle,lego,drone,robot,vacuum cleaner,iron,sewing machine,ladder,rope,bucket,"
        "broom,hose,paint roller"
    ).split(","),
    "documents_office": (
        "letter,envelope,stamp,notebook,calendar,whiteboard,projector,printer,scanner,"
        "keyboard,mouse,monitor,speaker,headphones,microphone,camera,lens,tripod,"
        "folder,binder,clipboard,pen,pencil,eraser,ruler,scissors,tape,box,package,"
        "barcode,label,gift tag,sticker,menu,ticket,boarding pass,menu board,sign,signpost,"
        "billboard,white paper,spreadsheet,slideshow,video call,server rack,hard drive"
    ).split(","),
    "art_media": (
        "photo,color photo,panorama,collage,mosaic,graffiti,mural,sculpture,sketch,"
        "watercolor,illustration,flyer,magazine,cover,stage,calligraphy,origami,"
        "pottery,vase,carpet pattern,tile pattern,embroidery,knitting,stained glass,"
        "neon sign,christmas tree,halloween,easter eggs,fireworks,parade,festival"
    ).split(","),
    "activities_weather_time": (
        "festival,trophy,medal,gym,yoga,skiing,snowboard,surfing,cycling,hiking,"
        "camping,tent,campfire,fishing,picnic blanket,sunrise yoga,morning,afternoon,"
        "evening,dawn,dusk,midnight,winter morning,rainy day,snowy day,windy day,"
        "foggy day,sunny day,cloudy day,harvest,picnic area,dance,marathon,chess,"
        "tennis,basketball,volleyball,baseball,golf,swimming pool,playground slide"
    ).split(","),
    "misc": (
        "pill,bandage,laboratory,microscope,telescope,globe,compass,binoculars,anchor,"
        "lifebuoy,saddle,horseshoe,traffic cone,construction site,roadworks,detour,"
        "toll booth,roundabout,escalator,elevator,revolving door,turnstile,lock,chain,"
        "padlock,safe,vault,atm,cash,coins,piggy bank"
    ).split(","),
}

BUILTIN = list(dict.fromkeys(_OLD_BUILTIN + [t for group in _EXTRA.values() for t in group]))

_LINE = re.compile(r"[\n;,\t]+")


def _normalized(values):
    seen, out = set(), []
    for raw in values:
        t = re.sub(r"\s+", " ", str(raw).strip().strip('"\'').replace('_', ' ')).strip().lower()
        if t and len(t) <= 60 and 'http' not in t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _normalized_report(values):
    """(kept, dropped): dropped counts long/url/empty/dup without silent loss."""
    seen, kept, dropped = set(), [], 0
    for raw in values:
        t = re.sub(r"\s+", " ", str(raw).strip().strip('"\'').replace('_', ' ')).strip().lower()
        if not t or len(t) > 60 or 'http' in t or t in seen:
            dropped += 1
            continue
        seen.add(t)
        kept.append(t)
    return kept, dropped


def parse_tags_report(text, top=2000):
    """parse_tags + dropped count (long/url/empty/dup). Additive; parse_tags unchanged."""
    import csv
    import io
    text = str(text).lstrip('\ufeff')
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if rows:
        header = [x.strip().lower() for x in rows[0]]
        name = next((header.index(k) for k in ('tag', 'name') if k in header), None)
        count = next((header.index(k) for k in ('count', 'post_count', 'frequency') if k in header), None)
        if name is not None:
            rows = [r for r in rows[1:] if len(r) > name]
            if count is not None:
                def frequency(row):
                    try:
                        return float(row[count])
                    except (ValueError, IndexError):
                        return 0
                rows.sort(key=frequency, reverse=True)
            kept, dropped = _normalized_report(r[name] for r in rows)
            kept = kept if top is None else kept[:max(1, min(int(top), 5000))]
            # Slicing by top can drop more; report them as dropped too.
            total_kept, _ = _normalized_report(r[name] for r in rows)
            dropped = len(total_kept) - len(kept) + dropped
            return {'tags': kept, 'dropped': dropped}
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(('#', '```')):
            continue
        lines.append(re.sub(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)', '', line))
    kept, dropped = _normalized_report(_LINE.split('\n'.join(lines)))
    kept = kept if top is None else kept[:max(1, min(int(top), 5000))]
    return {'tags': kept, 'dropped': dropped}


def parse_tags(text, top=2000):
    """Header CSV (optionally ranked), Markdown bullets, or a free-form list."""
    import csv
    import io
    text = str(text).lstrip('\ufeff')
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=',;\t')
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if rows:
        header = [x.strip().lower() for x in rows[0]]
        name = next((header.index(k) for k in ('tag', 'name') if k in header), None)
        count = next((header.index(k) for k in ('count', 'post_count', 'frequency') if k in header), None)
        if name is not None:
            rows = [r for r in rows[1:] if len(r) > name]
            if count is not None:
                def frequency(row):
                    try:
                        return float(row[count])
                    except (ValueError, IndexError):
                        return 0
                rows.sort(key=frequency, reverse=True)
            values = _normalized(r[name] for r in rows)
            return values if top is None else values[:max(1, min(int(top), 5000))]
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(('#', '```')):
            continue
        lines.append(re.sub(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)', '', line))
    values = _normalized(_LINE.split('\n'.join(lines)))
    return values if top is None else values[:max(1, min(int(top), 5000))]


def _safe_url(url):
    import ipaddress
    import socket
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password:
        raise ValueError('only public HTTPS URLs are allowed')
    if parts.hostname.lower() == 'github.com':
        bits = parts.path.split('/')
        if len(bits) >= 6 and bits[3] == 'blob':
            parts = parts._replace(netloc='raw.githubusercontent.com', path='/' + '/'.join(bits[1:3] + bits[4:]))
    addresses = socket.getaddrinfo(parts.hostname, parts.port or 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0].split('%')[0]).is_global for a in addresses):
        raise ValueError('private or non-public host is forbidden')
    return urlunsplit(parts)


def _download(url):
    import urllib.request as request
    from urllib.parse import urljoin
    class Redirect(request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            safe = _safe_url(urljoin(req.full_url, newurl))
            return super().redirect_request(req, fp, code, msg, headers, safe)
    safe = _safe_url(url)
    # Use urlopen with a validated redirect handler; tests can replace urlopen.
    opener = request.build_opener(request.ProxyHandler({}), Redirect())
    req = request.Request(safe, headers={'User-Agent': 'LindaProbe/1.1'})
    limit = 25 * 1024 * 1024
    # Pass the opener through a small local adapter, without changing global urllib state.
    with urlopen(req, timeout=15, opener=opener) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('tag source exceeds 25 MB')
    return data.decode('utf-8-sig', errors='replace')


def urlopen(req, timeout=15, opener=None):
    """Injectable transport, including the redirect policy."""
    return opener.open(req, timeout=timeout)


def fetch_tags(source, top=1000):
    """Return {tags, source, truncated}; source is text, path, URL, or an API dict."""
    top = max(1, min(int(top), 5000))
    spec = source if isinstance(source, dict) else {}
    top = max(1, min(int(spec.get('top', top)), 5000))
    if spec.get('preset'):
        if spec['preset'] != 'danbooru-general':
            raise ValueError('unknown preset')
        ranked = []
        for page in range(1, (top + 999) // 1000 + 1):
            url = ('https://danbooru.donmai.us/tags.json?search[category]=0&search[order]=count&limit=1000&page=' + str(page))
            rows = json.loads(_download(url))
            ranked.extend((r['name'], int(r.get('post_count', 0))) for r in rows)
        ranked.sort(key=lambda r: -r[1])
        tags, dropped = _normalized_report(r[0] for r in ranked)
        label = 'danbooru-general'
    else:
        value = (spec.get('url') or spec.get('text', '')) if spec else str(source)
        if not isinstance(value, str):
            raise ValueError('tag source must be text')
        label = 'text'
        if spec.get('url') or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', value):
            label = _safe_url(value)
            value = _download(label)
        elif not spec:
            try:
                path = Path(value)
                if path.is_file():
                    if path.stat().st_size > 25 * 1024 * 1024:
                        raise ValueError('tag source exceeds 25 MB')
                    value, label = path.read_text(encoding='utf-8-sig'), str(path)
            except OSError:
                pass
        rep = parse_tags_report(value, None)
        tags, dropped = rep['tags'], rep['dropped']
    capped = tags[:top]
    return {'tags': capped, 'source': label, 'truncated': len(tags) > top,
            'dropped': dropped + (len(tags) - len(capped))}


def _path(store_path):
    return Path(str(store_path) + ".tags.json")


def load(store_path):
    try:
        return json.loads(_path(store_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"tags": [], "assign": {}}


def _save(store_path, data):
    from uuid import uuid4
    target = _path(store_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def _tag_prompt(tag):
    """Zero-shot prompt per tag. Borrowed from sist2-script-clip (TAG_TEMPLATE
    'a photo of a {}'): the indefinite article measurably helps the English
    CLIP text encoder (tune-64 top-1 51->53, holdout/full precision+recall up,
    one-shot verified). Applied to ASCII tags only - Cyrillic tags keep
    'a photo of {}' (RU check 65->64, i.e. no gain, avoid English grammar
    inside Russian prompts)."""
    t = str(tag or '')
    if t and all(ord(c) < 128 for c in t):
        return 'a photo of a ' + t
    return 'a photo of ' + t


def _embed_tag_text(engine, tags, text):
    """Embed tag prompts, honouring text='auto'/multilingual. Returns (matrix, info).

    text: 'auto' (default) = multilingual module when the dictionary needs it
      (needs_multilingual_text over the joined tags), else the engine default;
      'standard'|'multilingual' force the choice. Unknown engine helpers
      (test fakes without _active_text) fall back to engine.embed_text.
    info: {text_module: standard|multilingual|engine-default, text_warning}.
    Never raises for a missing multilingual model: falls back with a warning.
    """
    import warnings as _w
    info = {"text_module": "engine-default", "text_warning": ""}
    try:
        from localdocs_imgsearch import needs_multilingual_text as _need
        want_multi = (text == "multilingual" or
                      (text == "auto" and _need(" ".join(tags))))
        if text not in ("auto", "standard", "multilingual"):
            raise ValueError("text must be auto/standard/multilingual")
    except ValueError:
        raise
    except Exception:
        want_multi = (text == "multilingual")
    if not want_multi:
        return np.asarray(engine.embed_text([_tag_prompt(t) for t in tags]), np.float32), info
    encoder = None
    try:
        get_text = getattr(engine, "_active_text", None)
        if callable(get_text):
            encoder = get_text()
    except Exception:
        encoder = None
    if encoder is None:
        _w.warn("multilingual text module unavailable; using standard tag text",
                RuntimeWarning)
        info["text_warning"] = "multilingual-unavailable"
        return np.asarray(engine.embed_text([_tag_prompt(t) for t in tags]), np.float32), info
    try:
        from localdocs_imgsearch import _MultilingualText
        if isinstance(encoder, _MultilingualText):
            info["text_module"] = "multilingual"
            return np.asarray(encoder.encode([_tag_prompt(t) for t in tags]), np.float32), info
    except Exception:
        pass
    try:
        info["text_module"] = "multilingual"
        return np.asarray(encoder.encode([_tag_prompt(t) for t in tags]), np.float32), info
    except Exception:
        try:
            info["text_module"] = "multilingual-delegate"
            return np.asarray(encoder.embed_text([_tag_prompt(t) for t in tags]), np.float32), info
        except Exception as exc:
            _w.warn("multilingual tag text failed (%s); using standard" % (exc,), RuntimeWarning)
            info["text_warning"] = "multilingual-failed"
            return np.asarray(engine.embed_text([_tag_prompt(t) for t in tags]), np.float32), info


def tag_store(engine, store_path, tags, per_image=3, min_score=0.2, folder=None, should_stop=None, only_paths=None, *, text='auto'):
    from localdocs_imgsearch import _INFERENCE_LOCK
    with _INFERENCE_LOCK, _TAG_LOCK:
        return _tag_store(engine, store_path, tags, per_image, min_score, folder, should_stop, only_paths, text)


def _tag_store(engine, store_path, tags, per_image, min_score, folder, should_stop, only_paths, text='auto'):
    """Label every picture of the store (optionally only those under `folder`). Returns {images, tagged}."""
    tags = list(dict.fromkeys(tags))
    if len(tags) > 5000:
        raise ValueError("The tag list exceeds 5000 entries; choose a smaller list")
    if not tags:
        raise ValueError("no tags")
    per_image = max(1, min(int(per_image), 20))
    min_score = max(-1.0, min(float(min_score), 1.0))
    paths, _, _, vectors, key = engine._load(store_path)
    engine._prepare()
    if key != engine._key:
        raise ValueError("Active model changed; reindex the folder")
    root = os.path.normcase(os.path.abspath(folder)) if folder else None
    selected = {os.path.normcase(os.path.abspath(p)) for p in only_paths} if only_paths is not None else None
    selected_dirs = [p for p in selected if os.path.isdir(p)] if selected is not None else []
    previous = load(store_path)
    assign, tagged, images = dict(previous.get('assign', {})), 0, 0
    indices = []
    for i, p in enumerate(paths):
        canonical = os.path.normcase(os.path.abspath(p))
        def inside(directory):
            try:
                return os.path.commonpath([canonical, directory]) == directory
            except ValueError:
                return False
        if selected is not None and canonical not in selected and not any(inside(d) for d in selected_dirs):
            continue
        if root and not inside(root):
            continue
        indices.append(i)
    # run_paths never computes a giant all-store score matrix for a small selection.
    vec = vectors[indices].astype(np.float32)
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
    text_mat, text_info = _embed_tag_text(engine, tags, text)
    text = np.asarray(text_mat, np.float32)
    text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-9)
    scores = vec @ text.T
    for i, row in zip(indices, scores):
        if should_stop and should_stop():
            break
        p = paths[i]
        top = np.argsort(-row)[:per_image]
        picked = [[tags[j], round(float(row[j]), 4)] for j in top if row[j] >= min_score]
        assign[p] = picked
        images += 1
        tagged += bool(picked)
    data = {"tags": list(dict.fromkeys(previous.get('tags', []) + tags)), "assign": assign,
            "dicts": dict(previous.get('dicts', {}) or {})}
    data["dicts"][folder or 'default'] = list(tags)
    _save(store_path, data)
    out = {"images": images, "tagged": tagged, "tags": len(tags)}
    out.update(text_info)
    return out


def search_tags(store_path, wanted, mode="any", limit=200):
    """Pictures carrying the wanted tags: mode any/all; sorted by best score."""
    data = load(store_path)
    wanted = [w.strip().lower() for w in wanted if w and w.strip()]
    out = []
    for p, picked in data.get("assign", {}).items():
        got = {t: s for t, s in picked}
        hits = [got[w] for w in wanted if w in got]
        if hits and (mode != "all" or len(hits) == len(wanted)):
            out.append({"path": p, "score": max(hits), "tags": [t for t, _ in picked]})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def tag_counts(store_path):
    counts = {}
    for picked in load(store_path).get("assign", {}).values():
        for t, _ in picked:
            counts[t] = counts.get(t, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])
