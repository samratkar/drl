---
layout: default
title: DRL Course Index
---

<!-- Reuse the styles from your main site index -->
<style>
  .tag-item, .category-item {
    cursor: pointer !important;
    transition: all 0.2s ease;
    user-select: none;
  }
  .tag-item:hover, .category-item:hover {
    background-color: #f3e8ff;
    transform: translateX(4px);
  }
  .tag-item.active, .category-item.active {
    background-color: #9b87f5;
    color: white;
    font-weight: 600;
  }
  .hidden-row {
    display: none !important;
  }
  .side-panel {
    position: sticky;
    top: 2rem;
  }
</style>

<div class="flex flex-col md:flex-row gap-8">
  
  <!-- Left Sidebar: Filtering -->
  <aside class="w-full md:w-64 flex-shrink-0">
    <div class="side-panel space-y-8">
      
      <!-- Search -->
      <div>
        <h3 class="text-sm font-bold text-gray-900 uppercase tracking-wider mb-3">Search</h3>
        <input type="text" id="artSearch" placeholder="Search titles..." 
               class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none"
               onkeyup="filterByText()">
      </div>

      <!-- Categories -->
      <div>
        <h3 class="text-sm font-bold text-gray-900 uppercase tracking-wider mb-3">Categories</h3>
        <div id="categoryList" class="space-y-1">
          <div class="category-item px-3 py-2 rounded-md text-sm active" onclick="clearFilters(this)">
            All Categories
          </div>
          {% assign categories = "" | split: "," %}
          {% for page in site.pages %}
            {% if page.category %}
              {% unless categories contains page.category %}
                {% assign categories = categories | push: page.category %}
              {% endunless %}
            {% endif %}
          {% endfor %}
          {% assign categories = categories | sort %}
          {% for cat in categories %}
            <div class="category-item px-3 py-2 rounded-md text-sm text-gray-600" onclick="filterByCategory('{{ cat }}', this)">
              {{ cat }}
            </div>
          {% endfor %}
        </div>
      </div>

      <!-- Tags -->
      <div>
        <h3 class="text-sm font-bold text-gray-900 uppercase tracking-wider mb-3">Tags</h3>
        <div id="tagsList" class="flex flex-wrap gap-2">
          {% assign tags = "" | split: "," %}
          {% for page in site.pages %}
            {% if page.tags %}
              {% for tag in page.tags %}
                {% unless tags contains tag %}
                  {% assign tags = tags | push: tag %}
                {% endunless %}
              {% endfor %}
            {% endif %}
          {% endfor %}
          {% assign tags = tags | sort %}
          {% for tag in tags %}
            <span class="tag-item px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs" onclick="filterByTag('{{ tag }}', this)">
              {{ tag }}
            </span>
          {% endfor %}
        </div>
      </div>

    </div>
  </aside>

  <!-- Main Content: Table -->
  <div class="flex-1">
    <div class="overflow-x-auto border border-gray-100 rounded-xl shadow-sm">
      <table class="w-full text-left border-collapse">
        <thead>
          <tr class="bg-gray-50 border-b border-gray-100">
            <th class="px-4 py-3 text-xs font-bold text-gray-500 uppercase">Title</th>
            <th class="px-4 py-3 text-xs font-bold text-gray-500 uppercase">Category</th>
            <th class="px-4 py-3 text-xs font-bold text-gray-500 uppercase">Tags</th>
          </tr>
        </thead>
        <tbody id="articleBody">
          {% assign sorted_pages = site.pages | sort: "path" %}
          {% for p in sorted_pages %}
            {% if p.title and p.url != "/" and p.path contains ".md" %}
              <tr class="article-row border-b border-gray-50 hover:bg-purple-50 transition-colors" 
                  data-category="{{ p.category }}" 
                  data-tags="{{ p.tags | join: ',' }}">
                <td class="px-4 py-4">
                  <a href="{{ p.url | relative_url }}" class="text-purple-600 font-medium hover:underline">{{ p.title }}</a>
                  <p class="text-xs text-gray-400 mt-1">{{ p.path }}</p>
                </td>
                <td class="px-4 py-4 text-sm text-gray-500">
                  {{ p.category | default: "General" }}
                </td>
                <td class="px-4 py-4">
                  <div class="flex flex-wrap gap-1">
                    {% for tag in p.tags %}
                      <span class="px-2 py-0.5 bg-purple-50 text-purple-600 rounded-full text-get-xs" style="font-size: 0.65rem;">{{ tag }}</span>
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

<script>
  function filterByCategory(cat, el) {
    document.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
    el.classList.add('active');
    
    document.querySelectorAll('.article-row').forEach(row => {
      if (row.dataset.category === cat) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
  }

  function filterByTag(tag, el) {
    document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
    el.classList.add('active');
    
    document.querySelectorAll('.article-row').forEach(row => {
      const tags = row.dataset.tags.split(',');
      if (tags.includes(tag)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
  }

  function filterByText() {
    const q = document.getElementById('artSearch').value.toLowerCase();
    document.querySelectorAll('.article-row').forEach(row => {
      const title = row.querySelector('a').innerText.toLowerCase();
      if (title.includes(q)) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
  }

  function clearFilters(el) {
    document.querySelectorAll('.category-item, .tag-item').forEach(i => i.classList.remove('active'));
    el.classList.add('active');
    document.querySelectorAll('.article-row').forEach(row => row.classList.remove('hidden-row'));
    document.getElementById('artSearch').value = '';
  }
</script>
