#!/usr/bin/env python3
"""Generate themed dataset images via ComfyUI API using Flux.2 Klein models.

Handles prompt template expansion locally, queues to ComfyUI, retrieves images.
Supports resuming from where it left off.

Usage:
    # Start ComfyUI first, then:
    uv run scripts/generate_dataset_comfyui.py --theme nature --num-images 10000 --output datasets/nature-512

    # Use faster 4B model:
    uv run scripts/generate_dataset_comfyui.py --theme nature --model 4b --num-images 10000 --output datasets/nature-512

    # List available themes:
    uv run scripts/generate_dataset_comfyui.py --list-themes

    # Dry-run (no ComfyUI needed):
    uv run scripts/generate_dataset_comfyui.py --theme nature --num-images 0
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "9b": (
        "home_diffusion/flux-2-klein-9b-fp8.safetensors",
        "home_text_encoders/qwen_3_8b_fp8mixed.safetensors",
    ),
    "4b": (
        "home_diffusion/flux-2-klein-4b.safetensors",
        "home_text_encoders/qwen_3_4b.safetensors",
    ),
}


# ---------------------------------------------------------------------------
# Theme prompt templates
# ---------------------------------------------------------------------------

THEME_DESCRIPTIONS = {
    "abstract": "Neon particles, tubes, spirals, marble on dark backgrounds",
    "nature": "Diatoms, mineral thin-sections, satellite imagery, coral, frost crystals, pollen",
    "ukiyoe": "Waves, cherry blossoms, cranes, koi, Mt Fuji — flat color, bold outlines",
    "albums": "Psychedelic posters, glitch art, bold graphic design, vinyl grooves",
    "architecture": "Brutalist facades, spiral stairs, gothic vaults, Islamic tiles, repeating grids",
    "pixel": "16-bit RPG overworlds, sprite sheets, side-scrollers, limited palettes",
    "botanical": "Pressed flowers, scientific illustration, cross-sections, light backgrounds",
    "darkpsy": "B&W sacred geometry, glitch, biomechanical — neon accents, mixed polarity",
}

THEME_TEMPLATES = {}

# --- Abstract neon (original) ---
THEME_TEMPLATES["abstract"] = r"""{scattered glowing {cyan|magenta|amber|emerald|violet|crimson|gold|white} particles floating in dark void, bokeh circles, embers, sparse dots, deep black background|vertical {cyan|magenta|amber|emerald|violet|crimson|gold|white} neon tubes on black, parallel glowing lines, abstract barcode, laser beams, digital rain, vertical light painting|horizontal {cyan|magenta|amber|emerald|violet|crimson|gold|white} and {cyan|magenta|amber|emerald|violet|crimson|gold|white} bands on black, layered neon stripes, abstract equalizer bars, glowing horizontal lines, color spectrum bands|{cyan|magenta|amber|emerald|violet|crimson|gold|white} geometric tessellation pattern, Voronoi cells, hexagonal tiling, hard edges, repeating polygons, fills entire frame|concentric {cyan|magenta|amber|emerald|violet|crimson|gold|white} rings on black, water ripple circles, expanding waves, circular target pattern, hypnotic|{cyan|magenta|amber|emerald|violet|crimson|gold|white} branching lightning bolts on black, fractal dendrites, river delta from above, forking electric arcs, thin glowing lines|polished {cyan|magenta|amber|emerald|violet|crimson|gold|white} and {cyan|magenta|amber|emerald|violet|crimson|gold|white} marble texture on dark background, agate cross-section, mineral veins, dense swirling stone pattern, no focal point, fills frame|{cyan|magenta|amber|emerald|violet|crimson|gold|white} spiral vortex on black, galaxy arm, whirlpool, single fibonacci spiral, rotational motion|shattered {cyan|magenta|amber|emerald|violet|crimson|gold|white} glass fragments on black, kaleidoscope shards, broken mirror, sharp angular pieces, prismatic|smooth {cyan|magenta|amber|emerald|violet|crimson|gold|white} to {cyan|magenta|amber|emerald|violet|crimson|gold|white} gradient on dark background, color field painting, minimal abstract, soft transition, no edges, blurred atmosphere|glowing {cyan|magenta|amber|emerald|violet|crimson|gold|white} circuit board traces on black, synthwave wireframe grid, electronic pathways, rectangular lines, neon matrix|single {cyan|magenta|amber|emerald|violet|crimson|gold|white} smoke plume rising on black, ink drop in water, flowing asymmetric form, wispy tendrils|{cyan|magenta|amber|emerald|violet|crimson|gold|white} radial starburst, explosion of light rays from center, sun corona, firework burst, radiating lines on black|cluster of {cyan|magenta|amber|emerald|violet|crimson|gold|white} iridescent soap bubbles, foam texture, packed spheres, translucent membranes, cellular|woven {cyan|magenta|amber|emerald|violet|crimson|gold|white} and {cyan|magenta|amber|emerald|violet|crimson|gold|white} interlocking abstract pattern, braided mesh, chainmail, repeating knot, textile texture|scattered {cyan|magenta|amber|emerald|violet|crimson|gold|white} crystal fragments floating on black, broken gemstone pieces, debris field, irregular shapes, sparse}, {high contrast, sharp focus|dark background, vivid color|detailed texture, crisp|atmospheric, moody|clean, minimal|bold, saturated}"""

# --- Nature macro/microscopy ---
THEME_TEMPLATES["nature"] = r"""{diatom under microscope, intricate {golden|emerald|sapphire|amber|pearl|copper} silica skeleton, radial symmetry, dark field microscopy|mineral thin section under polarized light, {amethyst|tourmaline|olivine|feldspar|quartz|garnet} crystal cross section, vivid interference colors, birefringent|satellite view of {river delta|coral reef|volcanic crater|salt flats|glacier|desert dunes}, false color {blue|orange|green|violet|teal|red} imagery, aerial topography|{brain|staghorn|fan|mushroom|table|elkhorn} coral closeup, {pink|orange|peach|lavender|cyan|cream} calcium carbonate structure, underwater macro photography|frost crystals on glass, {hexagonal|dendritic|fernlike|needle|plate|columnar} ice formation, {blue|white|silver|pale violet|cyan|crystal clear} tones on dark background, macro|pollen grain under electron microscope, {spiky|smooth|ridged|perforated|latticed|geometric} surface texture, {golden|amber|sage|dusty rose|ochre|ivory} tones, SEM style|cross-section of {agate|geode|petrified wood|jasper|opal|labradorite}, concentric bands, {purple|blue|rust|emerald|amber|crimson} mineral layers, polished stone|lichen growing on {granite|bark|slate|sandstone|marble|basalt}, intricate {chartreuse|sage|rust|gold|teal|silver} branching patterns, macro closeup|{monarch butterfly|dragonfly|beetle|moth|damselfly|cicada} wing detail, iridescent {blue|green|gold|purple|copper|orange} microscopic scales, extreme macro|bioluminescent {jellyfish|plankton|deep sea fish|coral|algae|squid} glowing {blue|green|cyan|violet|turquoise|white} in dark water, underwater photography|{snowflake|salt|sugar|bismuth|pyrite|fluorite} crystal, geometric {hexagonal|cubic|orthorhombic|trigonal|monoclinic|fractal} growth pattern, {clear|blue|gold|violet|silver|rose} crystal on dark background|{tide pools|moss|lichen|fungal mycelium|dew drops on spider web|aurora borealis}, natural {green|blue|gold|purple|teal|amber} palette, organic abstract, nature closeup|{peacock feather|abalone shell|soap film|oil slick|beetle carapace|hummingbird throat} iridescence, {rainbow|blue-green|magenta-gold|violet-teal|copper-emerald|rose-cyan} shifting colors, macro detail|cell division under microscope, {blue|green|purple|orange|red|yellow} fluorescent staining, mitosis stages, dark field, scientific imaging|{tree rings|wood grain|bamboo cross-section|palm fiber|leaf venation|bark texture} natural pattern, {warm brown|golden|dark walnut|pale birch|amber|honey} organic tones, closeup texture|{nautilus shell|ammonite fossil|snail shell|conch|spiral galaxy|fibonacci fern} logarithmic spiral, {pearl|amber|cream|golden|copper|silver} natural geometry, mathematical nature}, {high magnification, detailed texture|dark background, vivid color|scientific accuracy, sharp focus|ethereal, dreamy|organic, natural|luminous, transparent}"""

# --- Ukiyo-e woodblock prints ---
THEME_TEMPLATES["ukiyoe"] = r"""{{great wave|turbulent ocean|crashing surf|river rapids|waterfall cascade|whirlpool} in ukiyo-e style, {indigo|prussian blue|teal|cerulean|navy|cobalt} water, white foam, bold outlines, woodblock print|cherry blossom {branch|tree|petals falling|grove|canopy|tunnel}, {pink|white|pale rose|magenta|coral|blush} flowers, {gold|red|black|indigo|green|vermillion} background, Japanese woodblock style|{crane|heron|egret|ibis|stork|peacock} in {flight|water|reeds|pine tree|moonlight|mist}, {white|silver|red-crowned|grey|golden|blue} plumage, ukiyo-e flat color, bold outlines|{koi fish|goldfish|carp|catfish|sea bream|blowfish} swimming in {pond|stream|waterfall pool|lotus pond|ocean|rice paddy}, {red|gold|white|black|orange|calico} scales, Japanese print style|Mount Fuji {at dawn|in snow|with clouds|at sunset|through pines|from lake}, {red|blue|purple|pink|orange|gold} sky, classic ukiyo-e landscape, Hokusai style|{bamboo forest|pine trees|maple leaves|wisteria|iris garden|plum blossoms}, {green|gold|red|purple|indigo|crimson} foliage, {fog|rain|snow|moonlight|wind|morning dew}, Japanese landscape|{full moon|crescent moon|setting sun} over {ocean|mountains|rice fields|pagoda|bridge|harbor}, {silver|gold|red|amber|pearl|copper} celestial body, night scene, ukiyo-e|{dragon|tiger|fox|phoenix|turtle|oni} in Japanese mythology style, {gold|red|blue|green|black|white} and {crimson|indigo|emerald|silver|vermillion|purple}, dynamic pose, woodblock|{lantern|torii gate|pagoda|castle|bridge|tea house} in {snow|rain|cherry blossoms|autumn leaves|fog|moonlight}, {red|vermillion|gold|black|indigo|white} architecture, ukiyo-e scene|{wave pattern|cloud pattern|scale pattern|chrysanthemum pattern|geometric lattice|fan pattern} filling frame, {indigo|red|gold|teal|black|white} repeating motif, Japanese textile design|{geisha with parasol|samurai in armor|kabuki actor|sumo wrestler|merchant with cart|monk with staff}, {red|purple|gold|blue|green|black} kimono and costume details, portrait, woodblock|stormy {sea|mountain pass|bamboo grove|city street|harbor|cliff}, {dark indigo|grey|charcoal|deep purple|steel blue|black} atmosphere, {rain|lightning|wind|snow|waves|mist}, dramatic ukiyo-e|{octopus|whale|turtle|crab|shrimp|sea dragon} in {ocean|waves|coral|seaweed|whirlpool|deep water}, {red|blue|grey|orange|purple|green} marine creature, Japanese style|{Nihonbashi bridge|Tokaido road|Sumida river|Yoshiwara|Edo castle|Asakusa temple} famous place, {crowded|empty|festive|snowy|rainy|twilight} scene, Hiroshige style|{chrysanthemum|peony|morning glory|camellia|lotus|iris} floral study, {white|pink|red|purple|yellow|blue} petals, {gold|green|black|indigo|cream|vermillion} accents, botanical ukiyo-e}, {flat color, bold outlines|woodblock print texture, visible grain|traditional Japanese color palette|minimal shading, graphic|decorative, ornamental|dramatic composition, asymmetric}"""

# --- Album art / concert posters ---
THEME_TEMPLATES["albums"] = r"""{psychedelic concert poster, swirling {neon pink|electric blue|acid green|hot orange|deep purple|golden yellow} patterns, {band logo|skull|third eye|mushroom|rising sun|mandala} centerpiece, 1960s San Francisco typography|glitch art, {RGB|CMYK|neon|pastel|monochrome|vaporwave} color channels split, digital corruption, scan lines, pixel sorting, databending aesthetic|vinyl record grooves extreme closeup, {iridescent|rainbow|black|gold|holographic|translucent} surface, concentric circles, light reflections, macro photography|album cover collage, {surreal|pop art|dada|constructivist|maximalist|punk} mixed media, {bold|muted|neon|earth tone|monochrome|pastel} palette, graphic design|{neon sign|marquee letters|billboard|LED display|gas station sign|motel sign} glowing {pink|blue|red|amber|green|white} on dark, night photography, urban|retro {synthwave|vaporwave|outrun|retrowave|cyberpunk|Miami Vice} landscape, {pink|purple|cyan|orange|magenta|blue} neon grid, sunset gradient, chrome text, 1980s aesthetic|{cassette tape|boombox|turntable|synthesizer|guitar pedal|mixing console} illustration, {bold flat color|neon outline|pop art|blueprint|x-ray|halographic} style, music equipment art|abstract {soundwave|frequency spectrum|oscilloscope trace|waveform|spectrogram|audio visualizer} in {cyan|magenta|gold|green|white|red} on black, music visualization, digital art|{skull|hand|eye|heart|lips|brain} in {x-ray|neon glow|chrome|melting|geometric low-poly|wireframe} style, {blue|pink|gold|green|red|purple} tones, dark background, album art|{shattered mirror|kaleidoscope|prism light|double exposure|motion blur|long exposure} abstract, {vivid|muted|high contrast|dreamy|saturated|split-tone} color, art photography|{op art|kinetic art|moire pattern|halftone dots|dot matrix|hatched lines} abstract, {black and white|red and blue|yellow and purple|cyan and magenta|green and pink|orange and blue} contrast, optical illusion|{torn paper|spray paint stencil|wheat paste layers|screen print|risograph print|letterpress} texture, {punk|hip-hop|electronic|jazz|metal|indie} aesthetic, {red|black|gold|white|neon green|earth tone} palette, DIY zine art|cosmic {nebula|supernova|black hole|aurora|meteor shower|eclipse} scene, {purple|blue|orange|pink|teal|red} deep space colors, album art, science fiction|{equalizer bars|mixing board faders|patch cables|drum machine pads|piano keys|guitar strings} abstract closeup, {neon|chrome|matte black|brushed aluminum|candy-colored|holographic} finish, studio aesthetic|{crowd silhouette|stage lights|spotlight beam|laser show|smoke machine|confetti burst} concert scene, {red|blue|purple|amber|white|multicolor} dramatic lighting, live music energy|geometric {impossible shape|Escher stairs|Penrose triangle|Klein bottle|fractal|tesseract} in {chrome|neon|marble|glass|obsidian|gold}, surreal album art, dark background}, {bold graphic design, high contrast|gritty texture, analog feel|clean vector, sharp edges|moody atmosphere, dark tones|vibrant, eye-catching|lo-fi, distressed}"""

# --- Architecture / brutalism ---
THEME_TEMPLATES["architecture"] = r"""{brutalist concrete {facade|tower|bridge|parking garage|museum|housing block}, raw {grey|charcoal|warm grey|cool grey|beige|anthracite} exposed concrete, geometric repetition, dramatic shadows|spiral staircase from {above|below|side angle|center|edge|between floors}, {marble|concrete|iron|wood|glass|stone} steps, {white|black|gold|red|teal|warm brown} railing, geometric spiral, architectural photography|gothic {vault|cathedral ceiling|rose window|flying buttress|cloister arcade|nave}, {limestone|sandstone|dark stone|marble|granite|slate} architecture, pointed arches, dramatic lighting|Islamic geometric {tilework|mosaic|muqarnas ceiling|arabesque panel|zellige wall|carved stone screen}, {turquoise|cobalt|gold|emerald|white|terracotta} and {white|gold|black|blue|cream|navy} pattern, intricate repetition|{modernist|art deco|bauhaus|mid-century|international style|deconstructivist} building facade, {glass|steel|concrete|copper|brick|aluminum} and {steel|glass|stone|wood|titanium|bronze} materials, clean lines|{fire escape|scaffold|crane|transmission tower|radio antenna|cooling tower} industrial structure, {rusty|painted|galvanized|weathered|black|orange} metal, geometric pattern against sky|Japanese {temple|zen garden|torii gates|castle|tea house|shrine}, {vermillion|black|natural wood|white|gold|grey} structure, {minimalist|traditional|serene|symmetric|layered|contemplative} composition|{Roman|Greek|Egyptian|Mayan|Angkor Wat|Persian} ancient {columns|temple|archway|pyramid|carved relief|courtyard}, weathered {sandstone|limestone|marble|granite|basalt|travertine}, classical architecture|{greenhouse|atrium|shopping galleria|train station hall|airport terminal|library reading room} with {glass|wrought iron|steel|timber|lattice|cable} {roof|dome|canopy|ceiling|skylight|vault}, interior looking up, structural pattern|{brick|cobblestone|terrazzo|herringbone parquet|hexagonal tile|stone paving} floor or wall pattern, {red|grey|white|mixed tone|weathered|glazed} surface, repeating texture filling frame|{reflecting pool|infinity pool|fountain|aqueduct|water tower|concrete dam} architectural water feature, {turquoise|deep blue|dark mirror|rippled|emerald|still} water, geometric concrete, minimalist|{power plant|factory interior|grain silos|water treatment plant|observatory dome|lighthouse} industrial architecture, {concrete|steel|brick|corrugated metal|glass block|riveted iron} surfaces, monumental scale|{chapel of light|skylight shaft|window shadow|stained glass|light well|sun tunnel} architectural light study, {golden|white|colored|dappled|harsh|soft} light on {concrete|stone|wood|plaster|marble|brick}, chiaroscuro|{parking garage ramp|highway overpass|subway tunnel|pedestrian bridge|escalator|elevator shaft} utilitarian structure, {raw concrete|fluorescent lit|tiled|painted|bare|industrial} surfaces, geometric perspective|{Gaudi mosaic|art nouveau ironwork|baroque ceiling|rococo ornament|gothic tracery|Moorish arch} decorative architecture detail, {colorful|gold|white|polychrome|pastel|jewel-toned} ornament, intricate craftsmanship|{Manhattan skyline|Tokyo towers|Dubai skyline|Hong Kong density|favela hillside|brutalist housing estate} urban density, {sunset|night|foggy|stormy|blue hour|golden hour} atmosphere, architectural cityscape}, {dramatic shadows, high contrast|symmetrical composition, clean lines|moody atmosphere, overcast|warm golden hour light|minimal, geometric|raw texture, materiality}"""

# --- Pixel art / retro games ---
THEME_TEMPLATES["pixel"] = r"""{16-bit RPG overworld map, {lush green|autumn orange|snowy white|desert tan|volcanic red|tropical teal} tileset, {forest|mountain|ocean|castle|village|cave} terrain, retro game cartography|sprite sheet of {warrior|mage|robot|alien|dragon|knight} character, {walk cycle|idle animation|attack sequence|jump arc|run cycle|casting spell} frames, {NES 4-color|Game Boy green|SNES 16-color|GBA 32-color|CGA 4-color|Commodore 64} palette|side-scrolling platformer scene, {underground cave|cloud kingdom|enchanted forest|deep ocean|haunted castle|neon city} level, {blue|green|purple|orange|grey|pink} gradient sky, pixel art platforms|{dungeon|maze|spaceship corridor|ancient temple|magic library|science laboratory} interior, top-down view, {stone|metal|wood|crystal|bone|tech panel} walls, {torchlight|crystal glow|screen light|lava glow|mushroom light|neon} illumination, pixel art|{sword|shield|health potion|golden key|gemstone|magic scroll} game item, {gold|silver|emerald|ruby|sapphire|amethyst} coloring, clean pixel icon on dark background|{explosion|campfire|waterfall|smoke|magic sparkle|electric arc} effect sprite sheet, {red-orange|blue-white|blue-green|grey-white|purple-pink|yellow-white} palette, animation sequence, pixel art|{space invaders|asteroids|pac-man maze|falling tetrominos|breakout bricks|snake trail} inspired abstract pattern, {neon|pastel|monochrome|warm retro|cool blue|earth tone} limited palette, classic arcade aesthetic|{sunset|sunrise|northern lights|starfield|storm clouds|rainbow} pixel art skyline, {city|mountain range|ocean|dense forest|desert mesa|ancient ruins} silhouette, gradient dithering|{heart|star|coin|mushroom|flower|crown} repeating pixel icon grid, {red|gold|green|blue|pink|purple} and {white|black|yellow|cyan|cream|grey} two-tone pattern, tileable|{robot|slime monster|ghost|skeleton|bat swarm|evil wizard} pixel art character portrait, {menacing|cute|stoic|angry|surprised|sleeping} expression, bold simple shapes, large pixels|{treasure map|fantasy world map|star chart|circuit board|dungeon blueprint|procedural maze} pixel art, {parchment tan|dark navy|ocean blue|forest green|aged sepia|slate grey} background, detailed pixel illustration|{waterfall grotto|erupting volcano|crystal cavern|floating sky island|giant mushroom forest|underwater coral kingdom} pixel art landscape, {16-color|32-color|64-color} rich palette, detailed dithering|{8-bit city|medieval town|space station|pirate ship|dragon lair|enchanted garden} scene with {characters|vehicles|creatures|treasure|NPCs|enemies}, busy detailed pixel scene, retro RPG aesthetic|isometric {castle|factory|farm|dungeon|spaceship|town} pixel art, {stone grey|wood brown|metal blue|grass green|sand yellow|crystal purple} building blocks, {SimCity|Civilization|RollerCoaster|Habbo|Final Fantasy Tactics} inspired|{game over screen|title screen|character select|inventory menu|world map|shop interface} retro game UI, {blue|black|dark purple|red|green|grey} background, pixel text and icons, 8-bit style}, {clean pixel art, no anti-aliasing|limited color palette, retro charm|dithering patterns, textured|bold outlines, sharp pixels|nostalgic 1990s game aesthetic|chunky low resolution, charming}"""

# --- Botanical illustrations ---
THEME_TEMPLATES["botanical"] = r"""{pressed {rose|daisy|fern frond|lavender sprig|poppy|orchid} specimen, dried {green|brown|gold|sage|amber|olive} leaves and petals, white paper background, herbarium sheet, botanical study|scientific illustration of {lily|tulip|iris|sunflower|magnolia|peony} cross-section, {watercolor|ink|graphite pencil|gouache|pen and wash|colored pencil} on {cream|white|ivory|parchment|pale blue|light grey} paper, labeled parts|{succulent rosette|cactus|air plant|cushion moss|maidenhair fern|bromeliad} closeup, {jade green|dusty rose|sage|mint|coral|blue-green} tones, {white|cream|pale grey|soft pink|light blue|natural linen} background, botanical photography|{fly agaric mushroom|morel|chanterelle|bracket fungus|puffball|amanita} illustration, {red|brown|white|orange|yellow|purple} cap, detailed gills and stem, naturalist watercolor drawing|fruit cross-section of {pomegranate|fig|kiwi|passion fruit|dragonfruit|blood orange}, {ruby|purple|green|pink|white|golden} seeds and flesh, {white|cream|pale green|light blue|ivory|grey} background, food illustration|{seed pod|pine cone|acorn cluster|lotus pod|dandelion clock|milkweed burst} macro detail, {brown|gold|green|silver|cream|rust} natural texture, {white|cream|pale grey|natural linen|soft green|parchment} background, botanical study|{swallowtail butterfly|luna moth|jewel beetle|honeybee|ladybug|dragonfly} on {rose|daisy|lily|wildflower|thistle|clover}, {watercolor|gouache|colored pencil|oil paint|ink wash|scientific illustration} style, {cream|white|pale yellow|light green|soft blue|ivory} background|climbing {vine|rose|wisteria|ivy|honeysuckle|morning glory} pattern, {green|pink|purple|white|gold|red} foliage and blooms, decorative {border|wreath|repeating pattern|wallpaper design|textile print|garland} composition|{rosemary|sage|mint|lavender|thyme|basil} herb botanical plate, detailed {root system|stem cross-section|leaf structure|flower anatomy|seed formation|growth stages}, scientific illustration, labeled|{nautilus shell|sea urchin|starfish|branching coral|sand dollar|spiral seashell} natural specimen, {white|cream|pale pink|sandy|pearl|ivory} tones, natural history collection plate, light background|{peacock feather|spotted egg|woven nest|shed antler|animal skull|quartz crystal} natural object study, {soft warm|cool neutral|earth tone|muted|pale|natural} tones, {cream|white|grey|natural linen|pale blue|parchment} background, cabinet of curiosities|{wildflower meadow|tropical arrangement|dried flower bouquet|herb garden|succulent terrarium|woodland floor} botanical scene, {watercolor|oil|gouache|pencil|ink wash|pastel} technique, {white|cream|sage green|pale pink|light grey|ivory} background|{acorn|maple seed|pine nut|coconut|walnut|chestnut} botanical study, {cross-section|growth stages|germination|whole and halved|cluster|branch}, {brown|green|golden|cream|warm amber|russet} natural tones, scientific plate|{oak leaf|maple leaf|ginkgo|monstera|palm frond|eucalyptus} foliage study, {spring green|autumn gold|summer deep green|winter bare|bronze|silver-green} seasonal color, {watercolor|pressed specimen|pencil study|ink botanical|print|cyanotype} on light ground|{carnivorous plant|Venus flytrap|sundew|pitcher plant|orchid|corpse flower} exotic botanical, {green|red|purple|spotted|translucent|variegated} unusual form, detailed scientific illustration, {cream|white|pale green|black|grey|ivory} background|{wreath of herbs|garland of wildflowers|crown of roses|chain of daisies|ring of succulents|circle of mushrooms} circular botanical arrangement, {watercolor|ink|pencil|gouache|mixed media|digital botanical} style, {white|cream|pale sage|ivory|light grey|natural} background}, {scientific illustration, precise detail|soft watercolor washes, delicate|light airy background, breathing room|vintage naturalist style, aged paper|botanical accuracy, annotated|pressed and dried, archival quality}"""


# --- Darkpsy: B&W sacred geometry, glitch, biomechanical ---
THEME_TEMPLATES["darkpsy"] = r"""{sacred geometry mandala, {white on black|black on white|inverted|high contrast monochrome} bold thick lines, concentric circles and triangles, {neon green|acid green|UV purple|hot pink|electric blue|blood red} accent glow, geometric precision, hard edges|fractal pattern {Sierpinski triangle|Mandelbrot set|Koch snowflake|Menger sponge|Julia set|dragon curve}, {white on black|black on white|inverted|high contrast monochrome} sharp edges, {neon green|acid green|UV purple|hot pink|electric blue|blood red} accent highlights, mathematical recursion, bold graphic|glitch art pixel corruption, {white on black|black on white|inverted|high contrast monochrome} scan lines and data moshing, blocky digital artifacts, broken grid, harsh contrast, thick bands of noise|biomechanical texture {Giger|alien|organic machine|cybernetic|fused flesh and metal|mechanical organism} style, {white on black|black on white|inverted|high contrast monochrome} detailed surface, {neon green|acid green|UV purple|hot pink|electric blue|blood red} accent veins, bold sculptural forms|circuit board PCB traces, {white on black|black on white|inverted|high contrast monochrome} geometric pathways, {neon green|acid green|UV purple|hot pink|electric blue|blood red} solder points glowing, thick copper lines, electronic grid pattern|tribal mask totem face, {white on black|black on white|inverted|high contrast monochrome} bold carved features, thick geometric lines, angular symmetry, primitive blocky shapes, stark contrast|op art optical illusion, {white on black|black on white|inverted|high contrast monochrome} warped checkerboard, concentric distortion, moire pattern, bold graphic, thick repeating stripes|psychedelic third eye, {white on black|black on white|inverted|high contrast monochrome} radiating geometric rays, {neon green|acid green|UV purple|hot pink|electric blue|blood red} iris glow, bold sacred symbol, thick line art|corrupted data matrix code, {white on black|black on white|inverted|high contrast monochrome} falling digital characters, blocky pixel grid, harsh scan lines, broken typography, dense pattern|industrial metal grate, {white on black|black on white|inverted|high contrast monochrome} heavy diamond mesh pattern, thick welded bars, repeating grid, harsh shadows, utilitarian texture|X-ray medical scan, {white on black|black on white|inverted|high contrast monochrome} skeletal structure, {neon green|acid green|UV purple|hot pink|electric blue|blood red} diagnostic overlay, radiographic contrast, bold anatomical forms|UV blacklight paint splatter, {white on black|black on white|inverted|high contrast monochrome} base with {neon green|acid green|UV purple|hot pink|electric blue|blood red} fluorescent drips and splatters, bold chaotic marks, thick paint trails|noise static interference pattern, {white on black|black on white|inverted|high contrast monochrome} dense analog TV snow, blocky grain, harsh texture, uniform random pattern, fills entire frame|shattered glass fracture pattern, {white on black|black on white|inverted|high contrast monochrome} radiating crack lines, {neon green|acid green|UV purple|hot pink|electric blue|blood red} accent shards, bold geometric breakage, sharp angular fragments|organic branching {tentacles|mycelium network|dendrite growth|root system|lightning tree|coral branches}, {white on black|black on white|inverted|high contrast monochrome} thick forking lines, spreading fractal pattern, bold organic geometry|geometric tessellation {hexagonal grid|Penrose tiling|Voronoi cells|triangular mesh|Islamic star pattern|cubic lattice}, {white on black|black on white|inverted|high contrast monochrome} repeating polygons, {neon green|acid green|UV purple|hot pink|electric blue|blood red} accent edges, bold thick outlines, fills entire frame}, {high contrast, sharp focus|stark monochrome, graphic|bold lines, geometric|gritty texture, raw|minimal color, maximum impact|dark atmosphere, intense}"""

# ---------------------------------------------------------------------------
# Prompt template expansion (handles nested {a|b|c} syntax)
# ---------------------------------------------------------------------------

def expand_template(template: str) -> str:
    """Expand a dynamic prompt template by randomly picking from {a|b|c} groups."""
    result = template
    max_iter = 50
    for _ in range(max_iter):
        match = re.search(r'\{([^{}]+)\}', result)
        if not match:
            break
        options = match.group(1).split('|')
        choice = random.choice(options)
        result = result[:match.start()] + choice + result[match.end():]
    return result.strip()


# ---------------------------------------------------------------------------
# ComfyUI API-format workflow
# ---------------------------------------------------------------------------

def build_prompt(text: str, seed: int, model_key: str = "9b",
                 width: int = 512, height: int = 512) -> dict:
    """Build a ComfyUI API-format prompt dict."""
    unet_name, clip_name = MODELS[model_key]
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "flux2",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "home_vae/flux2-vae.safetensors",
            },
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": text,
            },
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["4", 0],
            },
        },
        "6": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 1.0,
            },
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },
        "8": {
            "class_type": "Flux2Scheduler",
            "inputs": {
                "steps": 4,
                "width": width,
                "height": height,
            },
        },
        "9": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["9", 0],
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["3", 0],
            },
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "glyph-dataset/img",
                "images": ["12", 0],
            },
        },
    }


# ---------------------------------------------------------------------------
# ComfyUI websocket client
# ---------------------------------------------------------------------------

class ComfyUIClient:
    def __init__(self, server: str = "127.0.0.1:8188"):
        self.server = server
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect(self):
        import websocket as ws_module
        self.ws = ws_module.WebSocket()
        self.ws.connect(f"ws://{self.server}/ws?clientId={self.client_id}")

    def close(self):
        if self.ws:
            self.ws.close()

    def queue_prompt(self, prompt: dict) -> str:
        prompt_id = str(uuid.uuid4())
        payload = {"prompt": prompt, "client_id": self.client_id, "prompt_id": prompt_id}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"http://{self.server}/prompt", data=data)
        urllib.request.urlopen(req).read()
        return prompt_id

    def wait_for_completion(self, prompt_id: str):
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                if msg["type"] == "executing":
                    d = msg["data"]
                    if d["node"] is None and d["prompt_id"] == prompt_id:
                        break
                elif msg["type"] == "execution_error":
                    d = msg["data"]
                    if d.get("prompt_id") == prompt_id:
                        raise RuntimeError(
                            f"ComfyUI execution error: {d.get('exception_message', 'unknown')}"
                        )

    def get_history(self, prompt_id: str) -> dict:
        url = f"http://{self.server}/history/{prompt_id}"
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read())

    def clear_history(self):
        """Clear ComfyUI history to prevent memory bloat on long runs."""
        try:
            data = json.dumps({"clear": True}).encode("utf-8")
            req = urllib.request.Request(f"http://{self.server}/history", data=data)
            urllib.request.urlopen(req).read()
        except Exception:
            pass

    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        url = f"http://{self.server}/view?{params}"
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    def generate_and_save(self, prompt: dict, output_path: Path, index: int) -> Path:
        """Queue prompt, wait, download image, save with sequential name."""
        prompt_id = self.queue_prompt(prompt)
        self.wait_for_completion(prompt_id)

        history = self.get_history(prompt_id)[prompt_id]
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                for img_info in node_output["images"]:
                    img_data = self.get_image(
                        img_info["filename"], img_info["subfolder"], img_info["type"]
                    )
                    out_file = output_path / f"{index:05d}.png"
                    out_file.write_bytes(img_data)
                    return out_file
        raise RuntimeError(f"No image output found for prompt {prompt_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_themes():
    """Print all available themes with descriptions."""
    print("Available themes:\n")
    for name, desc in THEME_DESCRIPTIONS.items():
        print(f"  {name:15s} {desc}")
    print(f"\n{len(THEME_DESCRIPTIONS)} themes available.")


def main():
    theme_list = "\n".join(f"  {k:15s} {v}" for k, v in THEME_DESCRIPTIONS.items())
    model_list = "\n".join(f"  {k:5s} {v[0]}" for k, v in MODELS.items())

    epilog = f"""
themes:
{theme_list}

models:
{model_list}

examples:
  %(prog)s --list-themes
  %(prog)s --theme nature --num-images 0          # dry-run, print sample prompts
  %(prog)s --theme nature --num-images 10000 --output datasets/nature-512
  %(prog)s --theme pixel --model 4b --num-images 5000
"""

    parser = argparse.ArgumentParser(
        description="Generate themed dataset via ComfyUI API",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--theme", choices=list(THEME_TEMPLATES.keys()),
                        help="Theme for prompt generation (see list below)")
    parser.add_argument("--model", choices=["9b", "4b"], default="9b",
                        help="Flux model (default: 9b, see list below)")
    parser.add_argument("--num-images", type=int, default=10000,
                        help="Total images to generate (0 for dry-run)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for images")
    parser.add_argument("--server", default="127.0.0.1:8188",
                        help="ComfyUI server address")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Clear ComfyUI history every N images")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset (useful for parallel generation)")
    parser.add_argument("--list-themes", action="store_true",
                        help="List available themes and exit")
    args = parser.parse_args()

    if args.list_themes:
        list_themes()
        return 0

    if not args.theme:
        parser.error("--theme is required (use --list-themes to see options)")

    template = THEME_TEMPLATES[args.theme]

    # Default output dir based on theme
    if args.output is None:
        args.output = Path(f"datasets/{args.theme}-512")

    # Dry-run: print sample prompts and exit
    if args.num_images == 0:
        print(f"Theme: {args.theme}")
        print(f"Model: {args.model} ({MODELS[args.model][0]})")
        print(f"Output: {args.output}")
        print(f"\nSample prompts:\n")
        for i in range(5):
            random.seed(i * 42)
            sample = expand_template(template)
            print(f"  [{i+1}] {sample}")
            print()
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    # Find where to resume from
    existing = sorted(args.output.glob("*.png"))
    start_idx = len(existing) + 1
    if start_idx > 1:
        print(f"Found {len(existing)} existing images, resuming from {start_idx:05d}")

    remaining = args.num_images - (start_idx - 1)
    if remaining <= 0:
        print(f"Already have {len(existing)} images (target: {args.num_images}). Done!")
        return 0

    print(f"Theme: {args.theme}")
    print(f"Model: {args.model} ({MODELS[args.model][0]})")
    print(f"Generating {remaining} images ({start_idx} to {args.num_images})")
    print(f"Output: {args.output}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Server: {args.server}")
    print()

    sample = expand_template(template)
    print(f"Sample prompt: {sample[:120]}...")
    print()

    # Connect to ComfyUI
    client = ComfyUIClient(args.server)
    try:
        client.connect()
    except Exception as e:
        print(f"ERROR: Cannot connect to ComfyUI at {args.server}")
        print(f"  {e}")
        print(f"\nMake sure ComfyUI is running:")
        print(f"  cd /home/kevin/git/comfyui && python main.py --listen --port 8188")
        return 1

    print("Connected to ComfyUI\n")

    t_start = time.time()
    errors = 0

    try:
        for i in range(start_idx, args.num_images + 1):
            prompt_text = expand_template(template)
            seed = (i + args.seed_offset) * 7919
            prompt = build_prompt(prompt_text, seed, args.model, args.width, args.height)

            t0 = time.time()
            try:
                client.generate_and_save(prompt, args.output, i)
            except Exception as e:
                errors += 1
                print(f"\n  ERROR on image {i}: {e}")
                if errors > 10:
                    print("Too many errors, stopping.")
                    break
                try:
                    client.close()
                    client.connect()
                except Exception:
                    pass
                continue

            elapsed = time.time() - t0
            total_elapsed = time.time() - t_start
            done = i - start_idx + 1
            rate = done / total_elapsed if total_elapsed > 0 else 0
            eta = (args.num_images - i) / rate if rate > 0 else 0

            if done % args.batch_size == 0:
                client.clear_history()
                print(f"\n  [batch {done // args.batch_size}] Cleared ComfyUI history")

            print(
                f"\r[{i:5d}/{args.num_images}] {elapsed:.1f}s/img | "
                f"{rate:.1f} img/s | ETA {eta/60:.0f}m | "
                f"errors={errors}",
                end="", flush=True,
            )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at image {i}. Resume by running the same command.")
    finally:
        client.close()

    total = time.time() - t_start
    final_count = len(list(args.output.glob("*.png")))
    print(f"\n\nDone! Generated {final_count} images in {total/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
