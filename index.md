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
    padding: 1.5rem;
    border: 1px solid #f3f4f6;
    shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  }
  .section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
</style>

<div class="flex flex-col lg:flex-row gap-6 items-start">
  
  <!-- LEFT PANEL: Search & Tags -->
  <aside class="w-full lg:w-72 flex-shrink-0">
    <div class="side-panel shadow-sm">
      
      <!-- Search -->
      <div class="mb-8">
        <h3 class="section-title">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          Search
        </h3>
        <input type="text" id="artSearch" placeholder="Filter by title..." 
               class="w-full px-3 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none transition-all"
               onkeyup="filterByText()">
      </div>

      <!-- Tags -->
      <div>
        <div class="flex items-center justify-between mb-3">
          <h3 class="section-title mb-0">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"></path></svg>
            Tags
          </h3>
          <button onclick="clearFilters()" class="text-[10px] text-purple-600 hover:text-purple-800 font-bold uppercase tracking-tighter">
            Clear All
          </button>
        </div>
        
        <div id="tagsList" class="flex flex-wrap gap-2">
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
            <span class="tag-item px-2 py-1 bg-gray-50 text-gray-600 rounded text-[11px] border border-gray-100" onclick="filterByTag('{{ tag }}', this)">
              {{ tag }}
            </span>
          {% endfor %}
        </div>
      </div>
    </div>
  </aside>

  <!-- MIDDLE PANEL: Table -->
  <div class="flex-1 min-w-0 w-full">
    <div class="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse">
          <thead>
            <tr class="bg-gray-50 border-b border-gray-100">
              <th class="px-6 py-4 text-[11px] font-bold text-gray-400 uppercase tracking-widest">Article Details</th>
              <th class="px-6 py-4 text-[11px] font-bold text-gray-400 uppercase tracking-widest">Category</th>
              <th class="px-6 py-4 text-[11px] font-bold text-gray-400 uppercase tracking-widest">Tags</th>
            </tr>
          </thead>
          <tbody id="articleBody">
            {% assign sorted_pages = site.pages | sort: "path" %}
            {% for p in sorted_pages %}
              {% if p.title and p.url != "/" and p.path contains ".md" and p.path != "index.md" and p.path != "README.md" %}
                <tr class="article-row border-b border-gray-50 hover:bg-purple-50/50 transition-colors" 
                    data-category="{{ p.category | default: 'General' }}" 
                    data-tags="{{ p.tags | join: ',' }}">
                  <td class="px-6 py-5">
                    <a href="{{ p.url | relative_url }}" class="text-gray-900 font-semibold hover:text-purple-600 transition-colors block">{{ p.title }}</a>
                    <span class="text-[10px] text-gray-400 font-mono mt-1 block">{{ p.path }}</span>
                  </td>
                  <td class="px-6 py-5">
                    <span class="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-1 rounded">{{ p.category | default: "General" }}</span>
                  </td>
                  <td class="px-6 py-5">
                    <div class="flex flex-wrap gap-1">
                      {% for tag in p.tags %}
                        <span class="px-2 py-0.5 bg-purple-50 text-purple-500 rounded text-[10px] font-medium border border-purple-100/50">{{ tag }}</span>
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

  <!-- RIGHT PANEL: Categories -->
  <aside class="w-full lg:w-64 flex-shrink-0">
    <div class="side-panel shadow-sm">
      <h3 class="section-title">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
        Categories
      </h3>
      <div id="categoryList" class="space-y-1">
        {% assign categories = "" | split: "," %}
        {% for p in site.pages %}
          {% if p.category %}
            {% unless categories contains p.category %}
              {% assign categories = categories | push: p.category %}
            {% endunless %}
          {% endif %}
        {% endfor %}
        {% assign categories = categories | sort %}
        
        <div class="category-item px-3 py-2 rounded-lg text-sm font-medium text-purple-600 bg-purple-50 active" onclick="clearFilters()">
          All Articles
        </div>
        
        {% for cat in categories %}
          <div class="category-item px-3 py-2 rounded-lg text-sm font-medium text-gray-600" onclick="filterByCategory('{{ cat }}', this)">
            {{ cat }}
          </div>
        {% endfor %}
      </div>
    </div>
  </aside>

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
