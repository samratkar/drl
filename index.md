---
layout: default
title: Deep Reinforcement Learning - A course on Barto Sutton's book
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
    background: white;
    border-radius: 0.75rem;
    padding: 1.25rem;
    border: 1px solid #e2e8f0;
    height: 100%;
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
  #lectureBody tr:nth-child(even) {
    background-color: #fcfcfd;
  }
</style>

<div class="w-full">

  <div class="flex flex-col xl:flex-row gap-6 items-stretch mb-8">
    
    <!-- LEFT PANEL: Tags -->
    <aside class="w-full xl:w-64 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <div class="flex items-center justify-between mb-4">
          <h3 class="section-title mb-0">Topic Tags</h3>
          <button onclick="clearFilters()" class="text-[9px] font-bold text-blue-600 hover:text-blue-800 uppercase tracking-widest transition-colors">Reset</button>
        </div>
        <div id="tagsList" class="flex flex-wrap gap-1.5">
          {% assign tags = "" | split: "," %}
          {% assign post_pages = site.pages | where: "layout", "post" %}
          {% for p in post_pages %}
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
    </aside>

    <!-- MIDDLE PANEL: Lectures Table -->
    <div class="flex-1 min-w-0">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden h-full">
        <div class="px-6 py-4 bg-slate-50/50 border-b border-slate-200 flex justify-between items-center">
          <h2 class="text-sm font-extrabold text-slate-700 uppercase tracking-widest m-0">Lectures</h2>
          <div class="relative w-48">
            <input type="text" id="artSearch" placeholder="Search..." 
                   class="w-full px-3 py-1 text-xs border border-slate-200 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all placeholder:text-slate-400"
                   onkeyup="filterByText()">
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse table-auto">
            <thead>
              <tr class="bg-slate-50/30 border-b border-slate-200">
                <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-12 text-center">#</th>
                <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Lecture Info</th>
                <th class="px-6 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Resources</th>
              </tr>
            </thead>
            <tbody id="lectureBody">
              {% assign lectures = site.pages | where: "layout", "post" | where: "category", "Lectures" %}
              {% for p in lectures %}
                {% assign filename = p.path | split: "/" | last | remove: ".md" %}
                {% assign parts = filename | split: "-" %}
                {% assign lecture_num = parts[0] | remove: "lecture" %}
                {% assign lecture_file = p.path | split: "/" | last %}
                {% assign lecture_dir = p.path | remove: lecture_file %}
                {% assign q_prefix = lecture_dir | append: "assets/questions/" %}
                {% assign s_prefix = lecture_dir | append: "assets/questions/solutions/" %}
                {% assign first_delivery = p.deliveries | first | default: "" %}
                
                <tr class="lecture-row border-b border-slate-100 hover:bg-blue-50/20 transition-colors" 
                    data-date="{{ first_delivery }}"
                    data-category="{{ p.subcategory | default: p.category }}"
                    data-tags="{{ p.tags | join: ',' }}"
                    data-num="{{ lecture_num }}">
                  <td class="px-6 py-4 text-sm font-black text-slate-400 text-center">{{ lecture_num }}</td>
                  <td class="px-6 py-4">
                    <div class="flex flex-col">
                      <a href="{{ p.url | relative_url }}" class="text-sm font-bold text-slate-800 hover:text-blue-600 transition-colors">{{ p.title }}</a>
                      <div class="flex items-center gap-2 mt-1">
                        {% if p.deliveries.size > 0 %}
                          {% for d in p.deliveries %}
                            <span class="text-[10px] text-slate-400 font-mono">{{ d }}</span>
                          {% endfor %}
                        {% else %}
                          <span class="text-[10px] text-slate-300 font-mono">TBD</span>
                        {% endif %}
                        <span class="text-[9px] font-bold text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-100">{{ p.subcategory | default: "General" }}</span>
                      </div>
                    </div>
                  </td>
                  <td class="px-6 py-4">
                    <div class="flex flex-col gap-2">
                      <div class="flex flex-wrap gap-1.5 items-center">
                        <span class="text-[9px] font-extrabold text-slate-400 uppercase tracking-tighter w-12">Problems</span>
                        {% assign found_questions = false %}
                        {% for q in site.pages %}
                          {% if q.layout == "post" and q.path contains q_prefix %}
                            {% unless q.path contains s_prefix %}
                              {% assign q_ext = q.path | split: "." | last %}
                              {% if q_ext == "md" %}
                                {% assign found_questions = true %}
                                {% assign q_name = q.path | split: "/" | last | remove: ".md" %}
                                <a href="{{ q.url | relative_url }}" class="text-[10px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded hover:bg-emerald-100 transition-colors no-underline border border-emerald-100">{{ q_name }}</a>
                              {% endif %}
                            {% endunless %}
                          {% endif %}
                        {% endfor %}
                        {% unless found_questions %}<span class="text-slate-300 text-[10px]">—</span>{% endunless %}
                      </div>
                      <div class="flex flex-wrap gap-1.5 items-center">
                        <span class="text-[9px] font-extrabold text-slate-400 uppercase tracking-tighter w-12">Solutions</span>
                        {% assign found_solutions = false %}
                        {% for s in site.pages %}
                          {% if s.layout == "post" and s.path contains s_prefix %}
                            {% assign s_ext = s.path | split: "." | last %}
                            {% if s_ext == "md" %}
                              {% assign found_solutions = true %}
                              {% assign s_name = s.path | split: "/" | last | remove: ".md" %}
                              <a href="{{ s.url | relative_url }}" class="text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded hover:bg-amber-100 transition-colors no-underline border border-amber-100">{{ s_name }}</a>
                            {% endif %}
                          {% endif %}
                        {% endfor %}
                        {% unless found_solutions %}<span class="text-slate-300 text-[10px]">—</span>{% endunless %}
                      </div>
                    </div>
                  </td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL: Focus Areas -->
    <aside class="w-full xl:w-56 flex-shrink-0">
      <div class="side-panel shadow-sm">
        <h3 class="section-title">Focus Areas</h3>
        <div id="categoryList" class="space-y-1">
          {% assign categories = "" | split: "," %}
          {% for p in lectures %}
            {% assign cat = p.subcategory | default: p.category %}
            {% if cat %}
              {% unless categories contains cat %}
                {% assign categories = categories | push: cat %}
              {% endunless %}
            {% endif %}
          {% endfor %}
          {% assign categories = categories | sort %}
          
          <div class="category-item px-3 py-2 rounded-md text-xs font-bold text-blue-600 bg-blue-50 border-blue-100 active" onclick="clearFilters(this)">
            All Lectures
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
    if (currentCategory === cat) {
      currentCategory = null;
      el.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
      el.classList.add('text-slate-500');
      document.querySelector('.category-item').classList.add('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
    } else {
      currentCategory = cat;
      document.querySelectorAll('.category-item').forEach(i => {
        i.classList.remove('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
        i.classList.add('text-slate-500');
      });
      el.classList.add('active', 'bg-blue-50', 'text-blue-600', 'border-blue-100');
      el.classList.remove('text-slate-500');
    }
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
    const rows = document.querySelectorAll('.lecture-row');

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
    
    document.querySelectorAll('.lecture-row').forEach(row => row.classList.remove('hidden-row'));
  }

  (function sortLectures() {
    const tbody = document.getElementById('lectureBody');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr.lecture-row'));
    rows.sort((a, b) => {
      const dateA = a.dataset.date || '';
      const dateB = b.dataset.date || '';
      if (dateA && dateB) return dateA.localeCompare(dateB);
      if (dateA) return -1;
      if (dateB) return 1;
      return parseInt(a.dataset.num) - parseInt(b.dataset.num);
    });
    rows.forEach(row => tbody.appendChild(row));
  })();
</script>
