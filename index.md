---
layout: default
title: Course Curriculum Index
---

<style>
  .tag-item, .category-item {
    cursor: pointer !important;
    transition: all 0.15s ease-in-out;
    user-select: none;
    border: 1px solid #e2e8f0;
  }
  .tag-item:hover, .category-item:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
    color: #0f172a;
  }
  .tag-item.active, .category-item.active {
    background-color: #2563eb !important;
    color: white !important;
    border-color: #1d4ed8 !important;
    font-weight: 600;
  }
  .hidden-row {
    display: none !important;
  }
  .side-panel {
    position: sticky;
    top: 2.5rem;
    background: white;
    border-radius: 0.75rem;
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
  }
  .section-title {
    font-size: 0.65rem;
    font-weight: 800;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .article-row:nth-child(even) {
    background-color: #fcfcfd;
  }
</style>

<div class="w-full">

  <!-- Lectures Section -->
  <div class="mb-8 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
    <div class="px-6 py-4 bg-slate-50/50 border-b border-slate-200">
      <h2 class="text-sm font-extrabold text-slate-700 uppercase tracking-widest m-0">Lectures</h2>
    </div>
    <div class="overflow-x-auto">
      <table class="w-full text-left border-collapse table-auto">
        <thead>
          <tr class="bg-slate-50/30 border-b border-slate-200">
            <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-16">#</th>
            <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Date</th>
            <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Title</th>
            <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Problems</th>
            <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Tags</th>
          </tr>
        </thead>
        <tbody id="lectureBody">
          {% assign lectures = site.pages | where: "category", "Lectures" %}
          {% for p in lectures %}
            {% assign filename = p.path | split: "/" | last | remove: ".md" %}
            {% assign parts = filename | split: "-" %}
            {% assign lecture_num = parts[0] | remove: "lecture" %}
            {% assign lecture_date = parts[1] | append: "-" | append: parts[2] | append: "-" | append: parts[3] %}
            {% assign lecture_file = p.path | split: "/" | last %}
            {% assign lecture_dir = p.path | remove: lecture_file %}
            {% assign q_prefix = lecture_dir | append: "assets/questions/" %}
            <tr class="border-b border-slate-100 hover:bg-blue-50/20 transition-colors" data-date="{{ lecture_date }}">
              <td class="px-6 py-4 text-sm font-bold text-slate-600">{{ lecture_num }}</td>
              <td class="px-6 py-4 text-sm text-slate-500 font-mono">{{ lecture_date }}</td>
              <td class="px-6 py-4">
                <a href="{{ p.url | relative_url }}" class="text-sm font-bold text-slate-800 hover:text-blue-600 transition-colors">{{ p.title }}</a>
              </td>
              <td class="px-6 py-4">
                <div class="flex flex-wrap gap-1.5">
                  {% assign found_questions = false %}
                  {% for q in site.pages %}
                    {% assign q_name = q.path | split: "/" | last %}
                    {% if q.path contains q_prefix %}
                      {% if q_name == "mcq.md" or q_name == "numerical.md" or q_name == "programming.md" %}
                        {% assign found_questions = true %}
                        <a href="{{ q.url | relative_url }}" class="text-sm font-bold text-emerald-700 hover:text-emerald-900 transition-colors no-underline">{{ q_name | remove: ".md" }}</a>
                      {% endif %}
                    {% endif %}
                  {% endfor %}
                  {% unless found_questions %}
                    <span class="text-slate-300 text-sm">—</span>
                  {% endunless %}
                </div>
              </td>
              <td class="px-6 py-4">
                <div class="flex flex-wrap gap-1.5">
                  {% for tag in p.tags %}
                    <span class="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[9px] font-bold border border-slate-200/50">{{ tag }}</span>
                  {% endfor %}
                </div>
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <div class="flex flex-col xl:flex-row gap-6 items-start">
    
    <!-- LEFT PANEL: Navigation & Tags -->
    <aside class="w-full xl:w-64 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <!-- Search -->
        <div class="mb-8">
          <h3 class="section-title">Filter Content</h3>
          <div class="relative">
            <input type="text" id="artSearch" placeholder="Type to search..." 
                   class="w-full pl-3 pr-3 py-2 text-sm border border-slate-200 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all placeholder:text-slate-400"
                   onkeyup="filterByText()">
          </div>
        </div>

        <!-- Tags -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <h3 class="section-title mb-0">Topic Tags</h3>
            <button onclick="clearFilters()" class="text-[9px] font-bold text-blue-600 hover:text-blue-800 uppercase tracking-widest transition-colors">Reset</button>
          </div>
          <div id="tagsList" class="flex flex-wrap gap-1.5">
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
              <span class="tag-item px-2 py-1 bg-slate-50 text-slate-600 rounded-md text-[10px] font-medium" onclick="filterByTag('{{ tag }}', this)">
                {{ tag }}
              </span>
            {% endfor %}
          </div>
        </div>
      </div>
    </aside>

    <!-- MIDDLE PANEL: Table -->
    <div class="flex-1 min-w-0 w-full">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse table-auto">
            <thead>
              <tr class="bg-slate-50/50 border-b border-slate-200">
                <th class="px-6 py-4 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-1/2">Module & Source Path</th>
                <th class="px-6 py-4 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-1/6 text-center">Focus Area</th>
                <th class="px-6 py-4 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-1/3">Tags</th>
              </tr>
            </thead>
            <tbody id="articleBody">
              {% assign sorted_pages = site.pages | sort: "path" %}
              {% for p in sorted_pages %}
                {% if p.title and p.url != "/" and p.path contains ".md" and p.path != "index.md" and p.path != "README.md" %}
                  <tr class="article-row border-b border-slate-100 hover:bg-blue-50/20 transition-colors" 
                      data-category="{{ p.category | default: 'Uncategorized' }}" 
                      data-tags="{{ p.tags | join: ',' }}">
                    <td class="px-6 py-5">
                      <a href="{{ p.url | relative_url }}" class="text-sm font-bold text-slate-800 hover:text-blue-600 transition-colors inline-block" title="{{ p.title }}">{{ p.title }}</a>
                      <div class="flex items-center gap-2 mt-1">
                        <span class="text-[9px] text-slate-400 font-mono tracking-tight">{{ p.path }}</span>
                      </div>
                    </td>
                    <td class="px-6 py-5 text-center">
                      <span class="text-[9px] font-black text-blue-700 bg-blue-50 px-2.5 py-1 rounded uppercase tracking-tighter border border-blue-100">{{ p.category | default: "General" }}</span>
                    </td>
                    <td class="px-6 py-5">
                      <div class="flex flex-wrap gap-1.5">
                        {% for tag in p.tags %}
                          <span class="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-[9px] font-bold border border-slate-200/50">{{ tag }}</span>
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
    <aside class="w-full xl:w-56 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <h3 class="section-title">Focus Areas</h3>
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
          
          <div class="category-item px-3 py-2 rounded-md text-xs font-bold text-blue-600 bg-blue-50 border-blue-100 active" onclick="clearFilters(this)">
            All Content
          </div>
          
          {% for cat in categories %}
            <div class="category-item px-3 py-2 rounded-md text-xs font-semibold text-slate-500" onclick="filterByCategory('{{ cat }}', this)">
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
    document.querySelectorAll('.category-item').forEach(i => {
      i.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
      i.classList.add('text-slate-500');
    });
    el.classList.add('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
    el.classList.remove('text-slate-500');
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

  function clearFilters(el = null) {
    currentTag = null;
    currentCategory = null;
    document.getElementById('artSearch').value = '';
    
    document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.category-item').forEach(i => {
      i.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
      i.classList.add('text-slate-500');
    });
    
    const allBtn = el || document.querySelector('.category-item');
    allBtn.classList.add('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
    allBtn.classList.remove('text-slate-500');
    
    document.querySelectorAll('.article-row').forEach(row => row.classList.remove('hidden-row'));
  }

  (function sortLectures() {
    const tbody = document.getElementById('lectureBody');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr[data-date]'));
    rows.sort((a, b) => a.dataset.date.localeCompare(b.dataset.date));
    rows.forEach(row => tbody.appendChild(row));
  })();
</script>
