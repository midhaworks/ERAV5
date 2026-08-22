# ERA V5 Assignment Branding & Sharing Standard

Use this checklist for every ERA V5 assignment page so branding, attribution, favicons, SEO, and social cards are correct before the first deployment.

## Required attribution

- Institution: **ERA V5 at The School of AI**
- Institution URL: <https://theschoolof.ai>
- Author wording: **By Avnish Midha**
- Author URL: <https://www.linkedin.com/in/avnishbm/>
- Keep the author credit subtle; do not say “Designed & built”.
- Adapt the session number to the assignment: `S4`, `S5`, `S6`, `S7`, `S8`, etc.

Recommended header treatment:

```html
<a href="https://theschoolof.ai" target="_blank" rel="noreferrer">
  ERA V5 · S8
</a>
```

Recommended footer hierarchy:

1. Assignment-specific source or accuracy note
2. `Session 8 · ERA V5 at The School of AI`
3. `View source on GitHub ↗`
4. `By Avnish Midha`

## Author and publisher metadata

```html
<meta name="author" content="Avnish Midha">
<meta property="article:author" content="https://www.linkedin.com/in/avnishbm/">
```

For article-like assignments, include a `TechArticle` JSON-LD block with:

- `headline`
- `description`
- absolute `image` URL
- truthful `datePublished` and `dateModified`
- `author` as Avnish Midha with the LinkedIn URL
- `publisher` as The School of AI with <https://theschoolof.ai>
- canonical `mainEntityOfPage`

Never let a crawler infer the publication date from dates inside the assignment content.

## Social-preview image

Create the card deliberately for social feeds rather than repurposing a screenshot.

- Dimensions: **1200 × 627 px**
- Preferred format for typography and diagrams: **PNG**
- Aim for less than **200 KB** when practical
- Use flat, high-contrast colours and large text
- Avoid paper texture, fine lines, small labels, and low-contrast detail
- Keep important content within generous safe margins
- Use a new versioned filename after meaningful revisions, e.g. `social-preview-v6.png`
- Serve the image from the same final production domain as the page
- Do not point production metadata at GitHub Raw or a temporary deployment host

Required metadata:

```html
<link rel="canonical" href="https://FINAL-DOMAIN/">
<link rel="image_src" href="https://FINAL-DOMAIN/assets/social-preview.png">

<meta property="og:type" content="article">
<meta property="og:url" content="https://FINAL-DOMAIN/">
<meta property="og:title" content="PAGE TITLE">
<meta property="og:description" content="PAGE DESCRIPTION">
<meta property="og:image" content="https://FINAL-DOMAIN/assets/social-preview.png">
<meta property="og:image:url" content="https://FINAL-DOMAIN/assets/social-preview.png">
<meta property="og:image:secure_url" content="https://FINAL-DOMAIN/assets/social-preview.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="627">
<meta property="og:image:alt" content="A concise description of the card">

<meta name="image" content="https://FINAL-DOMAIN/assets/social-preview.png">
<meta itemprop="image" content="https://FINAL-DOMAIN/assets/social-preview.png">

<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="PAGE TITLE">
<meta name="twitter:description" content="PAGE DESCRIPTION">
<meta name="twitter:image" content="https://FINAL-DOMAIN/assets/social-preview.png">
<meta name="twitter:image:alt" content="A concise description of the card">
```

Also add this namespace to the root element:

```html
<html lang="en" prefix="og: https://ogp.me/ns#">
```

## Favicon package

The favicon should be designed separately for tiny sizes. Do not shrink a detailed hero illustration blindly.

Provide all of these:

- Transparent SVG for modern browsers
- 32 × 32 transparent PNG for Safari and other fallbacks
- Root-level `/favicon.ico`
- 180 × 180 Apple touch icon
- Monochrome Safari pinned-tab SVG

Recommended markup:

```html
<link rel="shortcut icon" href="/favicon.ico?v=1">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/assets/favicon-32.png" type="image/png" sizes="32x32">
<link rel="apple-touch-icon" href="/assets/apple-touch-icon.png" sizes="180x180">
<link rel="mask-icon" href="/assets/safari-pinned-tab.svg" color="#5577ff">
```

For Attention Atlas, the chosen identity is:

- Transparent background
- Central `QKᵀ` core
- Clearly visible orbital paths
- One orange attention point
- No coordinate axes

Use a new filename or query version when changing favicons because Safari caches missing and old icons aggressively.

## SEO baseline

- Unique `<title>` and meta description
- Canonical production URL
- `index, follow, max-image-preview:large`
- Semantic single page-level `<h1>`
- Descriptive link text and image alt text
- Valid JSON-LD
- `robots.txt` that allows crawling
- `sitemap.xml` using the production URL
- No staging-domain URLs left in production metadata

## Pre-deployment crawler checks

Do not rely only on opening the page in a browser.

1. Fetch the HTML as `LinkedInBot/1.0` and `Twitterbot/1.0`.
2. Confirm the crawler receives all metadata in the initial HTML—not injected by JavaScript.
3. Fetch the absolute image URL with each crawler identity.
4. Require `200 OK`, the correct image MIME type, and expected dimensions.
5. Confirm the deployed image bytes match the intended local asset.
6. Test the final URL in LinkedIn Post Inspector.
7. Test WhatsApp and X separately because each platform maintains its own cache.

## Lessons from S8

- `workers.dev` loaded in browsers and WhatsApp but LinkedIn Post Inspector could not connect to it. Netlify worked.
- A technically valid textured JPEG looked blurred after LinkedIn thumbnailing. A flat vector-authored PNG remained crisp.
- LinkedIn may retain preview state inside an existing post draft. Test with a new draft after refreshing Post Inspector.
- Open Graph controls the supplied content, not LinkedIn's final card dimensions. For a guaranteed large LinkedIn visual, upload the card as a native image and place the URL in the post text.
- A native LinkedIn image is visually larger but is not itself a clickable link card.
- Safari may require **Show website icons in tabs**, clearing site data, and a full `⌘Q` restart after favicon changes.

## S8 production reference

- Live page: <https://attentionam.netlify.app/>
- Repository: <https://github.com/midhaworks/ERAV5/tree/main/S8>
- Working social card: `S8/assets/social-preview-v6.png`
- Working favicon family: `S8/assets/favicon-v5.svg`, `S8/assets/favicon-orbit-32.png`, and `S8/favicon.ico`
