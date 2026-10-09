Extract every meaningful durable conclusion supported by this source page. Do not
apply a top-k quota. Return complete=true only after considering the entire page.
If you cannot cover it within your output allowance, return complete=false and an
empty conclusions list; the caller will split and reprocess both halves.
Include explicit corrections, not tentative plans. This is partial context, not a
complete conversation. Later grouping and independent verification resolve it.
Use the supplied segment IDs even when the text is a fragment of that segment.
Each conclusion needs a 20-2000 character note, 1-8 exact user/tool quotes of
8-1200 characters each, and replaces=null. Never emit a partial JSON document.