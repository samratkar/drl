---
layout: default
title: DRL Course Index
---

<style>
  .tag-item, .category-item {
    cursor: pointer !important;
    transition: all 0.2s ease;
    user-select: none;
    border: 1px solid transparent;
  }
  .tag-item:hover, .category-item:hover {
    background-color: #f3e8ff;
    transform: translateX(4px);
    border-color: #c4b5fd;
  }
  .tag-item.active, .category-item.active {
    background-color: #9b87f5 !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 2px 4px rgba(155, 135, 245, 0.3);
  }
  .hidden-row {
    display: none !important;
  }
  .side-panel {
    position: sticky;
    top: 2rem;
    background: white;
    border-radius: 1rem;
    padding: 1.25rem;
    border: 1px solid #f3f4f6;
  }
  .section-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  /* Ensure the main container uses more horizontal space */
  .wide-container {
    max-width: 100% !important;
    width: 100%;
  }
</style>

<div class="wide-container px-4">
  <div class="flex flex-col xl:flex-row gap-4 items-start">
    
    <!-- LEFT PANEL: Search & Tags (Narrower) -->
    <aside class="w-full xl:w-56 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <!-- Search -->
        <div class="mb-6">
          <h3 class="section-title">Search</h3>
          <input type="text" id="artSearch" placeholder="Filter titles..." 
                 class="w-full px-3 py-2 text-xs border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none"
                 onkeyup="filterByText()">
        </div>

        <!-- Tags -->
        <div>
          <div class="flex items-center justify-between mb-3">
            <h3 class="section-title mb-0">Tags</h3>
            <button onclick="clearFilters()" class="text-[9px] text-purple-600 font-bold uppercase">Clear</button>
          </div>
          <div id="tagsList" class="flex flex-wrap gap-1">
            {% assign tags = "" | split: "," %}
            {% for p in site.pages %}
              {% if p.tags %}
                {% for tag in p.tags %}
                  {% unless tags contains tag %}
                    {% assign tags = tags | push: tag %}
                  {% endunless %}
                {% endfor %}
              {% endif %}
            {% endfor %}
            {% assign tags = tags | sort %}
            {% for tag in tags %}
              <span class="tag-item px-2 py-0.5 bg-gray-50 text-gray-500 rounded text-[10px] border border-gray-100" onclick="filterByTag('{{ tag }}', this)">
                {{ tag }}
              </span>
            {% endfor %}
          </div>
        </div>
      </div>
    </aside>

    <!-- MIDDLE PANEL: Table (Maximum Expansion) -->
    <div class="flex-1 min-w-0 w-full">
      <div class="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse table-auto">
            <thead>
              <tr class="bg-gray-50 border-b border-gray-100">
                <th class="px-4 py-3 text-[10px] font-bold text-gray-400 uppercase tracking-widest w-1/2">Article</th>
                <th class="px-4 py-3 text-[10px] font-bold text-gray-400 uppercase tracking-widest w-1/6">Category</th>
                <th class="px-4 py-3 text-[10px] font-bold text-gray-400 uppercase tracking-widest w-1/3">Tags</th>
              </tr>
            </thead>
            <tbody id="articleBody">
              {% assign sorted_pages = site.pages | sort: "path" %}
              {% for p in sorted_pages %}
                {% if p.title and p.url != "/" and p.path contains ".md" and p.path != "index.md" and p.path != "README.md" %}
                  <tr class="article-row border-b border-gray-50 hover:bg-purple-50/30 transition-colors" 
                      data-category="{{ p.category | default: 'General' }}" 
                      data-tags="{{ p.tags | join: ',' }}">
                    <td class="px-4 py-4">
                      <a href="{{ p.url | relative_url }}" class="text-sm font-semibold text-gray-800 hover:text-purple-600 block truncate" title="{{ p.title }}">{{ p.title }}</a>
                      <span class="text-[9px] text-gray-400 font-mono block mt-0.5">{{ p.path }}</span>
                    </td>
                    <td class="px-4 py-4">
                      <span class="text-[10px] font-bold text-purple-700 bg-purple-50 px-2 py-0.5 rounded-full border border-purple-100">{{ p.category | default: "General" }}</span>
                    </td>
                    <td class="px-4 py-4">
                      <div class="flex flex-wrap gap-1">
                        {% for tag in p.tags %}
                          <span class="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-[9px] font-medium">{{ tag }}</span>
                        {% endfor %}
                      </div>
                    </td>
                  </tr>
                {% endif %}
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL: Categories (Narrower) -->
    <aside class="w-full xl:w-48 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <h3 class="section-title">Categories</h3>
        <div id="categoryList" class="space-y-0.5">
          {% assign categories = "" | split: "," %}
          {% for p in site.pages %}
            {% if p.category %}
              {% unless categories contains p.category %}
                {% assign categories = categories | push: p.category %}
              {% endunless %}
            {% endif %}
          {% endfor %}
          {% assign categories = categories | sort %}
          
          <div class="category-item px-3 py-1.5 rounded-md text-xs font-medium text-purple-600 bg-purple-50 active" onclick="clearFilters()">
            All Articles
          </div>
          
          {% for cat in categories %}
            <div class="category-item px-3 py-1.5 rounded-md text-xs font-medium text-gray-500" onclick="filterByCategory('{{ cat }}', this)">
              {{ cat }}
            </div>
          {% endfor %}
        </div>
      </div>
    </aside>

  </div>
</div>

<script>
  let currentTag = null;
  let currentCategory = null;

  function filterByCategory(cat, el) {
    currentCategory = cat;
    document.querySelectorAll('.category-item').forEach(i => i.classList.remove('active', 'bg-purple-50', 'text-purple-600'));
    el.classList.add('active', 'bg-purple-50', 'text-purple-600');
    applyFilters();
  }

  function filterByTag(tag, el) {
    if (currentTag === tag) {
      currentTag = null;
      el.classList.remove('active');
    } else {
      currentTag = tag;
      document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
      el.classList.add('active');
    }
    applyFilters();
  }

  function filterByText() {
    applyFilters();
  }

  function applyFilters() {
    const searchText = document.getElementById('artSearch').value.toLowerCase();
    const rows = document.querySelectorAll('.article-row');

    rows.forEach(row => {
      const title = row.querySelector('a').innerText.toLowerCase();
      const cat = row.dataset.category;
      const tags = row.dataset.tags.split(',');

      const matchesText = title.includes(searchText);
      const matchesCat = !currentCategory || cat === currentCategory;
      const matchesTag = !currentTag || tags.includes(currentTag);

      if (matchesText && matchesCat && matchesTag) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
  }

  function clearFilters() {
    currentTag = null;
    currentCategory = null;
    document.getElementById('artSearch').value = '';
    document.querySelectorAll('.category-item, .tag-item').forEach(i => i.classList.remove('active', 'bg-purple-50', 'text-purple-600'));
    document.querySelector('.category-item').classList.add('active', 'bg-purple-50', 'text-purple-600');
    document.querySelectorAll('.article-row').forEach(row => row.classList.remove('hidden-row'));
  }
</script>
